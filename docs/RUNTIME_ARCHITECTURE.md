# Runtime architecture

Canonical inference path: `CascadeAlignAttProcessor` (`cascade/simulstream_processor.py`) wrapping a `CascadeSession` from `cascade/runtime.py`. Entry points at the project root:

| Entry point | Purpose |
|---|---|
| `run_simulstream_batch.py` | Streaming evaluation over one or many WAVs. **The canonical runner for quality/latency numbers.** |
| `run_simulstream_compare.py` | A/B two alignment backends sequentially on one clip. |
| `run_alignment_single_audio.py` | ASR-side diagnostic harness (seam comparison, repeat stability). |
| `run_mt_backend_parity.py` | Single-prompt or curated-set MT backend parity harness (subprocess-isolated). |
| `evaluate_cascade_outputs.py` | OmniSTEval + XCOMET-XL over a `run_simulstream_batch` artefact bundle. **Uses `.venv-evaluation`.** |

## Two independent runtime axes

### ASR / alignment backend

`CascadeRuntimeConfig.alignment_backend_name` selects the ASR frontend:

| Name | What it is | Stable? |
|---|---|---|
| `qwen_forced` | `Qwen3-ASR-1.7B` + `Qwen3-ForcedAligner-0.6B`, vLLM | ✅ default |
| `gemma_onepass_qk_fast` | Gemma-4-E4B ASR + audio AlignAtt `qk_fast` in one Transformers pass | ✅ |
| `gemma_vllm_qk_fast` | Gemma-4-E4B ASR via vLLM + engine-native audio observer | experimental |

All three produce `AlignmentResult(text, words)` with per-word end-times so downstream source normalization and the shipped punctuation-LCP commit path can operate on the same contract.

### MT backend

`CascadeRuntimeConfig.mt_backend_name` is fixed to the shipped translation backend:

| Name | What it is | Stable? |
|---|---|---|
| `gemma_vllm_alignatt` | Gemma-4-E4B MT via vLLM + engine-native MT observer (tensor buffers, `cudagraph=full`) | ✅ default |

Design details in [`MT_VLLM_BACKEND.md`](MT_VLLM_BACKEND.md).

### Current shipped surfaces

```
alignment_backend_name = "qwen_forced"
mt_backend_name        = "gemma_vllm_alignatt"
run_simulstream_batch default: chunk_ms=800, max_history_utterances=0
run_iwslt_submission presets: main_low_latency=850, main_high_latency=1500
```

See [`RESULTS.md`](RESULTS.md) for historical calibration numbers; some tables there come from a richer pre-simplification surface than the current worktree exposes.

## Module map (active source in `cascade/`)

```
cascade/runtime.py                         # CascadeRuntimeConfig, LoadedModelBundle, CascadeSession
cascade/simulstream_processor.py           # SimulStream SpeechProcessor wrapper
cascade/mt/base.py                         # BaseMTBackend + MT dispatcher / shared AlignAtt utilities
cascade/source_frontier.py                 # source accessibility frontier + word timestamp normalization
cascade/source_text.py                     # source text normalization for MT
cascade/text_surface.py                    # target text / incremental rendering
cascade/translation_variants.py            # prompt templates, rendered-prompt dataclass
cascade/emission.py                        # emission policy + delay registration
cascade/artifacts.py                       # output bundle schemas + writers

cascade/alignment/base.py                  # AlignmentBackend base + AlignmentResult
cascade/alignment/qwen_forced_backend.py   # qwen_forced
cascade/alignment/gemma_transformers_asr_backend.py
                                          # gemma_onepass_qk_fast
cascade/alignment/gemma_vllm_asr_backend.py
                                          # gemma_vllm_qk_fast (ASR observer)
cascade/alignment/gemma_vllm_asr_worker.py # gemma_vllm_qk_fast worker class

cascade/mt/gemma_vllm_backend.py           # gemma_vllm_alignatt MT backend
cascade/mt/gemma_vllm_observer.py          # MT observer module + reconstruction
cascade/mt/gemma_vllm_worker.py            # MT worker class

patch_qwen_asr_for_transformers5.py        # runtime monkey-patches for qwen_asr on Transformers 5
```

Historical compatibility shims and dated research scripts are intentionally
kept out of the active module map. See [`scripts/README.md`](../scripts/README.md)
if you need the legacy notebook/baseline helpers.

## Session + bundle lifecycle

```
CascadeRuntimeConfig                       # immutable-ish experiment config
   └── LoadedModelBundle                   # loads selected ASR + MT backends once
         ├── alignment_backend  (ASR)      # load() called lazily via ensure_alignment_backend()
         └── mt_backend         (MT)       # load() called lazily via ensure_mt_backend()
               └── CascadeSession          # mutable per-stream state; created via bundle.new_session()
                     ├── CascadeState      # utterance history, source, asr_hypotheses, utt_timestamps
                     ├── mt_prompt_cache   # PromptCacheState (currently inactive on the shipped MT path)
                     ├── partial_translation  # PartialTranslationState
                     └── streaming state   # ASR prefix carry-over state (experimental)
```

Bundle caching: `CascadeAlignAttProcessor._bundle_key(config)` includes `alignment_backend_name`, `mt_backend_name`, language pair, and heads path. Flipping ASR backends or MT engine-shaping knobs rebuilds the bundle cleanly.

## ASR-side commit path (current code)

The current worktree exposes a single ASR commit behaviour:

- **`punctuation_lcp` + EOS flush.** The runtime commits when the longest common prefix of two consecutive ASR hypotheses contains sentence-terminal punctuation, and `finalize_stream()` flushes the trailing tail even without a final punctuation cue.

Historical `alignatt_frontier` / `stable_and_accessible` ASR commit experiments are still documented in [`DECISIONS.md`](../DECISIONS.md) and [`RESULTS.md`](RESULTS.md), but they are not current runtime knobs and should be treated as archived calibration rather than active submission surfaces.

## ASR -> MT contract (shipped)

The shipped runtime intentionally separates **source conditioning** from
**target acceptance**:

- MT conditions on the **full live ASR tail** for the current sentence.
- Before prompting MT, the runtime strips only unstable trailing
  sentence-final punctuation from that live tail.
- `punctuation_lcp` still matters for **committed sentence history** and EOS
  flush, but it does **not** gate the partial MT call.
- AlignAtt is the sole runtime mechanism that limits how much new target text
  may be accepted and emitted from each partial MT draft.

This differs from `scripts/legacy_baseline.py`, whose target-side emission control is a
character-level local agreement between consecutive MT hypotheses.

## Latency/quality knob (today)

`--chunk-ms` is the main user-visible latency knob in the current worktree. The batch CLI defaults to `800`; the frozen submission presets use `850` and `1500`. Historical calibration on en→de shows:

- chunk 450 → ~1.7 s LongYAAL (CU), BLEU 27–28 en→de
- chunk 700 → ~3.5 s CA, BLEU ~31
- chunk 850 → ~4.7 s CA, BLEU ~37
- chunk 1500 → ~7.2 s CA, BLEU ~39

Numbers in [`RESULTS.md`](RESULTS.md). The larger ablation tables there include archived runs with knobs that are no longer public in the simplified runtime, so use each run manifest as the exact provenance when comparing old outputs.

## Extra-context (IWSLT sub-track) runtime axis

`CascadeRuntimeConfig.paper_context_mode ∈ {off, title_abstract, retrieved_chunks, title_and_chunks}` is an independent knob alongside the ASR/MT backend axes. Default `off` — every non-context caller is byte-identical to the pre-context runtime. When a `PaperArtifact` JSON is supplied via `paper_context_path`, the session computes a BM25 query from the current ASR prefix + recent source history and prepends a `[Paper context]` block to the Gemma MT user message, kept strictly outside the span tracked by `PromptSourceMap` so AlignAtt and the accepted-prefix contract remain intact. Pairs with `translation_alignatt_min_source_mass` (MT-AlignAtt provenance guard) to suppress paper-content leakage on close-to-talk papers. Full design: [`CONTEXT_INJECTION.md`](CONTEXT_INJECTION.md).

## See also

- [`MT_VLLM_BACKEND.md`](MT_VLLM_BACKEND.md) — MT observer / worker design, Phase 0–5 status
- [`RESULTS.md`](RESULTS.md) — consolidated quality/latency numbers
- [`CONTEXT_INJECTION.md`](CONTEXT_INJECTION.md) — extra-context mechanism, ablations, submission setting
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — operational gotchas
- `DECISIONS.md` (at repo root) — append-only session log
- `docs/archive/` — historical design notes preserved for context
- `docs/reference/` — upstream model cards and referenced papers/code
