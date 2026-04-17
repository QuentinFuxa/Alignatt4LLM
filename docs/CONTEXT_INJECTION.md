# MT-side extra-context injection

This document describes how the cascade injects paper-derived context into
the Gemma MT prompt for the IWSLT 2026 **Speech-to-Text with Extra Context**
sub-track. The mechanism is opt-in, default-off, and designed to be
defensible in a paper: no hand-written lexical substitutions, no
talk-specific prompt hacks, no hidden oracle information.

See `PLAN.md` for the 24h mission and `DECISIONS.md` for the Step 0 anchor
that pins Gemma MT (not Qwen ASR) as the context-injection substrate.

## Mechanism

```
ACL paper PDF
    └─ offline, deterministic
         └─→ PaperArtifact (JSON)
              ├─ title / authors / abstract
              └─ paragraph-level chunks (with section headers when recoverable)

at runtime, per MT call:
    query = recent source history (last N words) + current ASR prefix
    BM25 over artifact.chunks → top-k chunks under a fixed character budget
    render `[Paper context]` block with
        Title:     <title>
        Abstract:  <abstract>          (if the mode asks for it)
        [chunk_id | section]\n<chunk>  (if the mode asks for retrieval)
    prepend to the MT user message, *before* `[Confirmed earlier sentence pairs]`
    and well before `[Current <src> ASR prefix]`
```

The four supported modes are declared in
`context_injection.context_selector.VALID_CONTEXT_MODES`:

| Mode | What it renders |
|---|---|
| `off` | nothing. Runtime behaviour is bit-identical to pre-context. |
| `title_abstract` | front-matter only. Cheap baseline. |
| `retrieved_chunks` | title line + top-k BM25 chunks keyed to the live query. |
| `title_and_chunks` | title + abstract + top-k chunks. Richest. |

`retrieved_chunks` is the main contribution; the other modes are ablations.

## Defensibility

The retrieval scorer is transparent Okapi BM25 over a public per-paper
index. There is no hand-curated glossary, no LLM rerank, no benchmark-tuned
heuristic. The query uses only **already-confirmed** source text (current
ASR prefix + recent history) — no future audio, no references — so the
mechanism is consistent with the streaming time-of-use constraint.

The paper context is rendered *outside* the current-source span, and
`TranslationVariant._render_structured_user_message` raises if a caller
ever smuggles the source header into the paper block. That keeps
`PromptSourceMap` and AlignAtt operating on exactly the same current-source
tokens they did before, so the accepted-prefix contract is unchanged.

## Runtime knobs

Expressed on `CascadeRuntimeConfig` and surfaced on
`run_simulstream_batch.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--paper-context-path` | `None` | Path to a `PaperArtifact` JSON. Required when mode is not `off`. |
| `--paper-context-mode` | `off` | One of `off`, `title_abstract`, `retrieved_chunks`, `title_and_chunks`. |
| `--paper-context-top-k` | `3` | Max chunks retrieved per MT call. |
| `--paper-context-max-chars` | `1200` | Budget on the rendered block, before the `[Paper context]` header. |
| `--paper-context-history-window-words` | `60` | Words of confirmed source history to concatenate into the retrieval query. |

`max_chars` is a hard ceiling. BM25 retrieves in score order; a chunk that
would not fit is dropped cleanly rather than partial-truncated (except for
the top-ranked chunk, which may be word-boundary-truncated so the block is
never empty when retrieval found anything relevant).

## Producing paper artifacts

One talk ↔ one paper ↔ one artifact. Use the PDF extractor:

```bash
python -m context_injection.paper_artifact \
    test-set/pdf/ccpXHNfaoy.pdf test-set/pdf/OiqEWDVtWk.pdf \
    -o data/paper_artifacts
```

This reads each PDF with `pymupdf4llm`, drops headers/footers, splits front
matter on `/\nabstract\n/i`, strips everything from `References` onward, and
produces paragraph-level chunks (min `--min-chunk-chars`, max
`--max-chunk-chars`; defaults 200 / 1000). The output is a
schema-versioned JSON file per paper.

The extraction is deterministic: same PDF + same library versions → same
artifact bytes.

## Canonical reproducible commands

Baseline (no context), single clip:

```bash
python run_simulstream_batch.py \
    --wavs test-set/audio/OiqEWDVtWk.wav \
    --output-dir outputs/ctx_none_OiqEWDVtWk \
    --alignment-backend-name qwen_forced \
    --mt-backend-name gemma_transformers_alignatt \
    --chunk-ms 450 --source en --target de
```

Static title+abstract:

```bash
python run_simulstream_batch.py \
    --wavs test-set/audio/OiqEWDVtWk.wav \
    --output-dir outputs/ctx_titleabstract_OiqEWDVtWk \
    --paper-context-path data/paper_artifacts/OiqEWDVtWk.json \
    --paper-context-mode title_abstract \
    --chunk-ms 450 --source en --target de
```

Retrieved chunks (main mechanism):

```bash
python run_simulstream_batch.py \
    --wavs test-set/audio/OiqEWDVtWk.wav \
    --output-dir outputs/ctx_retrieved_OiqEWDVtWk \
    --paper-context-path data/paper_artifacts/OiqEWDVtWk.json \
    --paper-context-mode retrieved_chunks \
    --paper-context-top-k 3 --paper-context-max-chars 1200 \
    --chunk-ms 450 --source en --target de
```

Evaluate and compare the three bundles with
`evaluate_cascade_outputs.py` using `.venv-evaluation`.

## Invariants protected by tests

`test_context_injection.py` pins, without GPU:

- Paper artifact JSON round-trips through `read_json` / `write_json`.
- BM25 retrieval is deterministic and orders by query overlap (`MuST-C /
  English / German / BLEU` surfaces the evaluation chunk; `decoder /
  attention / emit` surfaces the method chunk).
- `render_messages` with `paper_context_block=""` is byte-identical to
  the call-site omitting the kwarg entirely — `off` mode is free.
- `render_messages` with a non-empty block puts `[Paper context]` before
  `[Current <src> ASR prefix]` and still reports the correct
  `source_text_char_span_in_user_message` (the substring at that span is
  the original source text).
- A paper block that includes the source header string is rejected at
  render time (would otherwise break `rfind(source_header)` in
  `build_prompt_source_map`).
- `CascadeRuntimeConfig` refuses `paper_context_mode != off` when no
  `paper_context_path` is provided; default `off` works with no path.

## First empirical sanity pass

On `tmp/ccpXHNfaoy_first75.wav` (75 s prefix of the Distilling-Script-Knowledge
talk), `qwen_forced` + `gemma_transformers_alignatt`, chunk_ms=450,
top_k=3, max_chars=1200 — same session, same hot bundle (ASR output
identical across all three conditions, so the deltas are MT-only):

| Mode | RTF | first_emit_audio_s |
|---|---|---|
| `off` | 1.512 | 4.05 |
| `title_abstract` | 1.440 | 4.05 |
| `retrieved_chunks` | 1.491 | 3.15 |

Qualitative findings on this clip (details in `DECISIONS.md`):

- `retrieved_chunks` corrects paper-specific terminology that `off` gets
  wrong ("distilling from" instead of "and", "procedural steps", paper-
  consistent constraint phrasing).
- `title_abstract` **leaks**: the static abstract example
  *"bake a cake for diabetics"* ends up inserted verbatim into the German
  translation even though the English ASR never said "diabetics". Same
  example does **not** leak under `retrieved_chunks` — paragraph-level
  chunks are less tempting for the model to quote than a tight abstract.

Retrieved chunks is the recommended setting. Static title+abstract is
kept as an ablation baseline, not a production mode.

## Paper-content leakage and the provenance guard

On a second clip (`tmp/OiqEWDVtWk_first90.wav`, AlignAtt paper) the
`retrieved_chunks` mode was observed to leak paper sentences verbatim
into the German translation ("EDATT", "Wait-k", "Local Agreement" —
none spoken in the English ASR). See `DECISIONS.md` for the full
analysis.

Two mitigations were tried:

1. **System-prompt reference-only clause** — caused Gemma-4-E4B to
   mode-collapse on retrieved_chunks (literal `}` runs,
   "That is that is…" loops). Rolled back; the slot on
   `TranslationVariant.paper_context_instruction_template` is kept
   `None` on the shipped variant but available for future attempts.
2. **Provenance guard via `translation_alignatt_min_source_mass`** —
   the MT AlignAtt observer already produces a 4-way partition
   (`source_accessible / source_inaccessible / non_source_prompt /
   suffix`). Setting the threshold above 0.0 vetoes any drafted token
   whose source-accessible attention mass is below the cut. The
   `[Paper context]` block counts as `non_source_prompt`, so
   paper-attending drafts get filtered out with stop reason
   `alignatt:provenance_weak`. **Partial win:** blatant leaks (EDATT,
   Wait-k) are eliminated at `0.3`, a weaker interpolation-paraphrase
   leak survives; no mode collapse, ~6 % RTF cost.

Recommended non-experimental pair when the mechanism is enabled:

```bash
python run_simulstream_batch.py \
    --wavs test-set/audio/OiqEWDVtWk.wav \
    --output-dir outputs/ctx_retrieved_guarded \
    --paper-context-path data/paper_artifacts/OiqEWDVtWk.json \
    --paper-context-mode retrieved_chunks \
    --paper-context-top-k 3 --paper-context-max-chars 1200 \
    --translation-alignatt-min-source-mass 0.3 \
    --chunk-ms 450 --source en --target de
```

Cross-clip confirmation (see `DECISIONS.md` for detail): the same
`min_source_mass=0.3` value holds on three clips (ccpXHNfaoy
Distilling Script Knowledge, OiqEWDVtWk AlignAtt, myfXyntFYL
Prompting PaLM). On every clip the guard removes visible paper leakage
from `title_abstract` and unlocks terminology wins ("Das Destillieren",
"SimulST-Modelle", "PaLM" + "540 Milliarden"). `retrieved_chunks` has
higher ceiling on one clip (ccpXHNfaoy) but retains subtle leaks on
the other two, so the **paper-ready default is `title_abstract +
min_source_mass=0.3`**. Latency cost is ~20 % RTF on context-on modes
relative to `off`.

## Non-goals

- No ASR-side term priming in the default build. The `--source` prompt
  reaching Qwen ASR is unchanged by anything in `context_injection/`.
- No cross-document retrieval: exactly one paper per talk.
- No embedding / LLM rerank. BM25 stays the scorer until we have empirical
  evidence that a heavier retriever moves the needle.
- No benchmark-tuned special cases. Everything in this pipeline runs the
  same code on every talk.
