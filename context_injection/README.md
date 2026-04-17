# context_injection/

MT-side extra-context preprocessing and runtime selection for the
IWSLT 2026 Speech-to-Text with Extra Context sub-track.

## Active modules

| File | Role |
|---|---|
| `paper_artifact.py` | **Canonical PDF → JSON extractor.** Produces a schema-versioned `PaperArtifact` (title / authors / abstract / paragraph chunks). Deterministic; no LLM. CLI: `python -m context_injection.paper_artifact <pdf>...`. |
| `context_selector.py` | **Runtime selector.** Okapi BM25 over chunks, budget-bounded `[Paper context]` block rendering. Four modes (`off`, `title_abstract`, `retrieved_chunks`, `title_and_chunks`). |
| `__init__.py` | Re-exports the public API. |

See [`../docs/CONTEXT_INJECTION.md`](../docs/CONTEXT_INJECTION.md) for
the mechanism design, empirical ablations, and the recommended
submission setting (retrieved chunks + provenance guard via
`translation_alignatt_min_source_mass`).

## Legacy bootstrap scripts

Kept for reference only — **nothing active in the runtime imports
them**. They pre-date the `PaperArtifact` schema and were copied from
the public IWSLT baseline pattern during initial exploration.

- `extract_abstract.py` — pre-`PaperArtifact` PDF-to-text bootstrap.
  Superseded by `paper_artifact.build_paper_artifact_from_pdf`.
- `ner_llm.py` — Qwen3-30B-A3B structured NER over a paper's
  title/authors/abstract. Potentially useful if a future
  entity-based context mechanism is explored, but not currently
  referenced by any runtime path.

If either script is revived, migrate its output into the
`PaperArtifact` schema rather than introducing a parallel artifact
shape.
