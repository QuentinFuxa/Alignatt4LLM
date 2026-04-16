# Results — Qwen3 ASR + Gemma vLLM MT cascade

This doc consolidates the end-to-end results produced in the 2026-04-16 Phase 0 → Phase 6 push. All numbers come from `run_simulstream_batch.py` using the real `CascadeAlignAttProcessor` path — no research-harness shortcuts. All evaluations go through `evaluate_cascade_outputs.py` (OmniSTEval + XCOMET-XL).

## Recommended runtime configuration

- `alignment_backend_name = "qwen_forced"`  (Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B)
- `mt_backend_name        = "gemma_vllm_alignatt"`  (Gemma-4-E4B MT through vLLM + engine-native AlignAtt observer)

Defaults that matter for latency/quality:

```
chunk_ms                                = 450  (see calibration table below)
min_start_seconds                       = 2.0
max_history_utterances                  = 1
partial_max_new_tokens                  = 16
partial_followup_max_new_tokens         = 8
translation_alignatt_rewind_threshold   = 8
translation_alignatt_inaccessible_ms    = 0     # has ~zero effect, see below
translation_alignatt_min_source_mass    = 0.0
mt_vllm_enable_prefix_caching           = False
mt_vllm_cudagraph_mode                  = "full"
mt_vllm_gpu_memory_utilization          = 0.5
```

## End-to-end SimulStream on `test-set/audio/ccpXHNfaoy.wav` (360 s, en→de)

Config above, baseline latency regime.

| Metric                | Value   |
|-----------------------|---------|
| BLEU                  | 27.51   |
| chrF                  | 63.54   |
| COMET (XCOMET-XL)     | 0.861   |
| LongYAAL (CU)         | 1766 ms |
| LongYAAL (CA)         | 1473 ms |
| RTF                   | 0.401   |
| Wallclock             | 144.5 s |
| Resegmented instances | 47      |
| Empty predictions     | 0       |

Artifacts: `outputs/simulstream_phase6_one_clip/`.

## Latency calibration curve on `test-set/audio/OiqEWDVtWk.wav` (299 s, en→de)

Controlled `--chunk-ms` sweep (same SHA, same machine, same config otherwise).

| chunk_ms | LongYAAL CA | LongYAAL CU | BLEU  | chrF  | COMET | RTF   |
|----------|-------------|-------------|-------|-------|-------|-------|
| 450      | 1650        | 1931        | 26.96 | 63.46 | 0.830 | 0.433 |
| **700**  | **3539**    | **3789**    | **31.25** | **66.97** | **0.889** | **0.365** |
| 850      | 4740        | 5024        | 36.95 | 68.83 | 0.914 | 0.335 |
| 1500     | 7169        | 7556        | 38.91 | 70.02 | 0.924 | 0.258 |

**Operating point for ~3500 ms LongYAAL (CA): `chunk_ms = 700`.**

Notes from the calibration:

- `--translation-alignatt-inaccessible-ms 2000` had effectively zero effect (1650 → 1650 CA, same BLEU). The cascade's scheduler already waits on commits/finals, so the inaccessibility mask does not bite the way it bites in a hypothetical pure-partial system. **Chunk size is the clean latency knob for this architecture.**
- Quality scales monotonically with chunk size over the tested range; COMET saturates around chunk 1500 (~0.92). LongYAAL CA scales roughly 7 ms per ms of chunk_ms over 450–850.
- RTF improves with bigger chunks because fewer MT calls per second amortize the vLLM prefill cost better.

## Phase-level validation history (same SHA, same machine where noted)

### Phase 5 — smoke18 (18 s, non-test-set clip; sanity only)

Config: `qwen_forced` + `gemma_vllm_alignatt`, `chunk_ms=450`.

- RTF 0.536, wallclock 9.64 s, 18 updates, no observer failures, 0 crashes.
- Hypothesis (final): *"Hallo, ich bin Siyu Yuan von der Fudan University. Ich bin hier, um unser Werk vorzustellen. Unterscheidung von Skriptwissen und großen Sprachmodellen. Für eingeschränkte Sprachplanung. Im Alltag. Menschen planen ihre Handlungen oft Schritt für Schritt."*
- Artifact: `outputs/simulstream_qwen_forced_mt_vllm_smoke18/`.

### Phase 4 — curated single-prompt parity (no SimulStream loop)

Harness: `run_mt_backend_parity.py --prompt-set tmp/mt_parity_set.json` with **subprocess isolation per backend** (PyTorch + vLLM allocators cannot share a CUDA context safely). Six prompts chosen to exercise: final / no-prefill / with-prefill / provenance-weak / long-frontier / long-with-prefill.

- Draft text equal: **6/6**
- Stop reason equal: **5/6** (the miss is `partial_with_prefill`, where vLLM hits `alignatt:rewind` and Transformers hits `alignatt:source_frontier` on a 1-token offset)
- Blocked-position equal: **5/6**
- Acceptance text exact: 3/6, within 3 tokens: 6/6
- Provenance mean within 0.05: 0/6 (numerical drift; see Phase 2 note below)

Aligned source positions agree on ~80–90 % of tokens (e.g. 12/13 on `partial_no_prefill`). Decision-level parity is much higher than provenance-magnitude parity because argmax is robust to sub-1 % shifts in softmax weights.

Artifacts: `tmp/mt_parity_curated.json`.

### Phase 2 — observer signal validation (single-prompt)

Both backends on a synthetic partial prompt:

- `"Hi I'm Si Yuan from Fudan University and I"` → both stop at `alignatt:source_frontier`, blocked local position 11, unit 8 ("I" is the inaccessible word).
- `"Today we are going to talk about the challenges of simultaneous speech translation and"` → both stop at `alignatt:source_frontier`, blocked position 13, unit 13 ("and").

Observer diagnostics: `effective_head_count=8`, `missing_heads=[]`, `prompt_capture_count==prompt_length`, `decode_q_count==n_generated` for every selected layer.

## Known numerical drift (Phase 2/3)

Provenance *magnitudes* differ between the Transformers MT backend (Python hook + qk_fast) and the vLLM MT backend (engine-native tensor observer). Argmaxes agree far more often than magnitudes because softmax argmax is robust to the small per-head numerical drift coming from:

- vLLM's **fused QKV projection** (`QKVParallelLinear.split(...)`) vs Transformers' **separate Q/K/V projections** (`q_proj`, `k_proj`, `v_proj`)
- vLLM's Gemma4-specific **proportional RoPE** kernel vs Transformers' `apply_rotary_pos_emb(partial_rotary_factor=0.25)` for full-attention layers
- Gemma 4 E4B has different head_dim per layer type (`head_dim=256` sliding, `global_head_dim=512` full); both backends handle this but through slightly different code

The numerical deltas are within single-digit ms per-token but accumulate across 8 heads and get amplified through softmax.

**Practical impact:** acceptance decisions (stop_reason, blocked_frontier) are reproducible; provenance *absolute means* are not bit-identical. For paper figures that depend on exact provenance numbers, use the Transformers backend. For runtime / latency measurements, the vLLM backend is the one we ship.

## Historical RTF anchors (for reference)

From `PLAN.md` / `DECISIONS.md` / earlier design notes, on `tmp/alignatt_smoke18.wav`:

| Backend combination                                     | RTF      | Source            |
|---------------------------------------------------------|----------|-------------------|
| `qwen_forced` + Transformers MT (stable baseline)       | 0.798    | PLAN snapshot     |
| `gemma_onepass_qk_fast` + Transformers MT               | 2.950    | PLAN snapshot     |
| `gemma_vllm_qk_fast` + Transformers MT                  | 2.305    | PLAN snapshot     |
| **`qwen_forced` + `gemma_vllm_alignatt` (Phase 5 new)** | **0.536** | this push        |

Caveat: smoke18 is a short clip, different git SHAs produced the snapshots, and the `0.536 ↔ 0.798` comparison hasn't been re-run same-code. Do not quote a specific speedup percent without a clean control run. The Phase 6 calibration table above is the reliable ground-truth for quality/latency; the smoke18 numbers exist only for sanity-check continuity with the pre-merge history.
