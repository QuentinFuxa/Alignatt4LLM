# Runtime architecture

Canonical inference path: `CascadeAlignAttProcessor` (`cascade_simulstream_processor.py`) wrapping a `CascadeSession` from `cascade_runtime.py`. Entry points at the project root:

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

All three produce `AlignmentResult(text, words)` with per-word end-times so the downstream AlignAtt-frontier commit rule (see below) can operate uniformly.

### MT backend

`CascadeRuntimeConfig.mt_backend_name` selects the translation backend:

| Name | What it is | Stable? |
|---|---|---|
| `gemma_transformers_alignatt` | Gemma-4-E4B MT via Transformers + Python-hook AlignAtt | ✅ default |
| `gemma_vllm_alignatt` | Gemma-4-E4B MT via vLLM + engine-native MT observer (tensor buffers, `cudagraph=full`) | experimental |

Design details in [`MT_VLLM_BACKEND.md`](MT_VLLM_BACKEND.md).

### Recommended combination today

```
alignment_backend_name = "qwen_forced"
mt_backend_name        = "gemma_vllm_alignatt"   # experimental, but Phase 5-validated end-to-end
```

See [`RESULTS.md`](RESULTS.md) for the numbers backing this recommendation.

## Module map (active source at repo root)

```
cascade_runtime.py                   # CascadeRuntimeConfig, LoadedModelBundle, CascadeSession
cascade_simulstream_processor.py     # SimulStream SpeechProcessor wrapper
cascade_mt_backend.py                # BaseMTBackend + TransformersAlignAttGemmaMTBackend + dispatcher
cascade_source_frontier.py           # source accessibility frontier + word timestamp normalization
cascade_source_text.py               # source text normalization for MT
cascade_text_surface.py              # target text / incremental rendering
cascade_translation_variants.py      # prompt templates, rendered-prompt dataclass
cascade_emission.py                  # emission policy + delay registration
cascade_artifacts.py                 # output bundle schemas + writers

alignment_backend.py                 # AlignmentBackend base + AlignmentResult
qwen_alignment_backend.py            # qwen_forced
gemma_alignment_probe.py             # gemma_onepass_qk_fast
gemma_vllm_alignment_backend.py      # gemma_vllm_qk_fast (ASR observer)
gemma_vllm_worker.py                 # gemma_vllm_qk_fast worker class

gemma_vllm_mt_backend.py             # gemma_vllm_alignatt MT backend
gemma_vllm_mt_observer.py            # MT observer module + reconstruction
gemma_vllm_mt_worker.py              # MT worker class

patch_qwen_asr_for_transformers5.py  # runtime monkey-patches for qwen_asr on Transformers 5
qwen3asr_gemma_cascade_core.py       # compatibility shim over the instantiable runtime; do not add features here
```

## Session + bundle lifecycle

```
CascadeRuntimeConfig                       # immutable-ish experiment config
   └── LoadedModelBundle                   # loads selected ASR + MT backends once
         ├── alignment_backend  (ASR)      # load() called lazily via ensure_alignment_backend()
         └── mt_backend         (MT)       # load() called lazily via ensure_mt_backend()
               └── CascadeSession          # mutable per-stream state; created via bundle.new_session()
                     ├── CascadeState      # utterance history, source, asr_hypotheses, utt_timestamps
                     ├── mt_prompt_cache   # PromptCacheState (Transformers MT only)
                     ├── partial_translation  # PartialTranslationState
                     └── streaming state   # ASR prefix carry-over state (experimental)
```

Bundle caching: `CascadeAlignAttProcessor._bundle_key(config)` includes `alignment_backend_name`, `mt_backend_name`, language pair, and heads path. Flipping backends rebuilds the bundle cleanly.

## AlignAtt-frontier commit rule (`asr_commit_mode="alignatt_frontier"`, default)

Replaces the earlier punctuation-LCP rule. Commit every contiguous prefix of words whose AlignAtt-aligned `end_time` is at least `asr_alignatt_frontier_margin_ms` (default 500 ms) behind the current audio frontier. Mono-mechanism: MT-side acceptance already uses the same frontier semantics.

Legacy rule preserved as `asr_commit_mode="punctuation_lcp"` for ablation. Details and empirical justification in `DECISIONS.md` section 2 (2026-04-16 session).

## Latency/quality knob (today)

`--chunk-ms` controls how often the system wakes up to emit. Empirically:

- chunk 450 → ~1.7 s LongYAAL (CU), BLEU 27–28 en→de
- chunk 700 → ~3.5 s CA, BLEU ~31
- chunk 850 → ~4.7 s CA, BLEU ~37
- chunk 1500 → ~7.2 s CA, BLEU ~39

Numbers in [`RESULTS.md`](RESULTS.md). `--translation-alignatt-inaccessible-ms` has ~zero effect in this architecture (the scheduler already waits on commits); `--translation-alignatt-min-source-mass` can add ~1 BLEU at a ~250 ms CU cost on en→de if desired.

## See also

- [`MT_VLLM_BACKEND.md`](MT_VLLM_BACKEND.md) — MT observer / worker design, Phase 0–5 status
- [`RESULTS.md`](RESULTS.md) — consolidated quality/latency numbers
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — operational gotchas
- `DECISIONS.md` (at repo root) — append-only session log
- `docs/archive/` — historical design notes preserved for context
- `docs/reference/` — upstream model cards and referenced papers/code
