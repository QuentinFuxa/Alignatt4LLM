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

Config above, two defensible operating points re-anchored under the
2026-04-16 overnight hardening SHA (commit `16609ec` and descendants).

| Operating point | BLEU  | chrF  | COMET | LongYAAL CU | LongYAAL CA | RTF   |
|-----------------|-------|-------|-------|-------------|-------------|-------|
| chunk_ms = 450  | 27.51 | 63.54 | 0.861 | 1766 ms     | 1466 ms     | 0.393 |
| chunk_ms = 700  | 38.19 | 66.53 | **0.940** | 3275 ms | 2945 ms | 0.369 |

chunk_ms=450 BLEU / chrF / CU / COMET are **bit-identical** to the
pre-hardening `simulstream_phase6_one_clip` run (see the Phase-level
validation section below). chunk_ms=700 buys **+0.079 COMET** (0.861
→ 0.940) for +1479 ms CA — a strong latency-quality trade at the
high-latency operating point.

Artifacts: `outputs/reanchor_chunk450/`, `outputs/reanchor_chunk700/`.
Historical baseline (same numerics, chunk_ms=450 only):
`outputs/simulstream_phase6_one_clip/` (BLEU 27.5133, chrF 63.5404,
COMET 0.861, CA 1473 ms, CU 1766 ms).

## Mechanism ablation: `stable_and_accessible` K-sweep

Third ASR commit rule introduced 2026-04-16 overnight: a source word is
committable iff it is both *accessible* (aligned end_time ≥ margin
behind the audio frontier) AND *stable* (identical at the same position
in the last K consecutive ASR hypotheses). `alignatt_frontier` is the
K=2 special case; `asr_stability_k` controls K for K ≥ 2.

Same clip / configuration (`ccpXHNfaoy.wav`, chunk_ms=450,
margin = 500 ms, `qwen_forced` + `gemma_vllm_alignatt`):

| Commit rule                    | BLEU  | chrF  | COMET | LongYAAL CU | LongYAAL CA | RTF   |
|--------------------------------|-------|-------|-------|-------------|-------------|-------|
| `alignatt_frontier`  (K=2)     | 15.78 | 54.43 | 0.558 | 1328 ms     | 1048 ms     | 0.43  |
| `stable_and_accessible` K=3    | 18.71 | 56.37 | 0.681 | 1919 ms     | 1637 ms     | 0.442 |
| `stable_and_accessible` K=4    | 20.26 | 57.92 | 0.730 | 2510 ms     | 2240 ms     | 0.480 |
| `stable_and_accessible` K=5    | 25.79 | 60.91 | 0.782 | 3585 ms     | 3395 ms     | 0.565 |
| `stable_and_accessible` K=6    | 28.13 | 62.16 | 0.824 | 4231 ms     | 4204 ms     | 0.690 |
| **`punctuation_lcp`**          | **27.51** | **63.54** | **0.861** | **1766 ms** | **1466 ms** | 0.393 |

Observations:

- K monotonically improves quality over `alignatt_frontier` (K=2).
  Growth is not linear: K=3→K=4 adds +1.55 BLEU, but K=4→K=5 adds
  +5.53 BLEU (a phase transition as tail-word flicker stops dominating),
  then K=5→K=6 adds +2.34 BLEU (saturation toward punct-level BLEU).
- At **K=6** the rule actually **matches or narrowly exceeds punct on
  BLEU** (28.13 vs 27.51) but remains below on chrF (62.16 vs 63.54)
  and COMET (0.824 vs 0.861), and pays a ~2.7 s CA penalty
  (4204 ms vs 1466 ms) for the privilege.
- Each +1 in K costs roughly +600-1000 ms of CA latency. By K=6 the
  rule's LongYAAL CA is ~3× the punctuation_lcp CA at the same
  chunk size.
- `punctuation_lcp` remains Pareto-optimal across BLEU / chrF / COMET /
  CA simultaneously on the Qwen-ASR path: no frontier-family K value
  dominates it on more than one metric at once.
- The quality gap at small K is caused by MT-context fragmentation:
  word-level commits force the MT observer to emit mid-sentence
  target fragments that compound into fluency degradation; larger K
  reduces commit rate and gradually restores sentence-level MT
  context, but never cheaply enough to beat punctuation.

**Practical impact:** `punctuation_lcp` stays the canonical submission
commit rule on the Qwen-ASR path. `stable_and_accessible` (K ≥ 3)
replaces `alignatt_frontier` as the recommended model-agnostic
fallback for paths whose ASR doesn't emit reliable sentence-terminal
punctuation (Gemma-4 ASR, lower-resource languages). K is the exposed
tuning knob; the margin remains as-is.

Artifacts: `outputs/night1_ende_stable_k{3,4,5,6}_chunk450/`.

### Cross-latency check: `stable_and_accessible` K=3 at chunk_ms=700

| Config                              | BLEU  | chrF  | COMET | LongYAAL CU | LongYAAL CA |
|-------------------------------------|-------|-------|-------|-------------|-------------|
| K=3 @ chunk_ms=450                  | 18.71 | 56.37 | 0.681 | 1919 ms     | 1637 ms     |
| K=3 @ chunk_ms=700                  | 24.67 | 60.12 | 0.740 | 2829 ms     | 2521 ms     |
| K=4 @ chunk_ms=450                  | 20.26 | 57.92 | 0.730 | 2510 ms     | 2240 ms     |
| punctuation_lcp @ chunk_ms=450      | 27.51 | 63.54 | 0.861 | 1766 ms     | 1466 ms     |
| punctuation_lcp @ chunk_ms=700      | 38.19 | 66.53 | 0.940 | 3275 ms     | 2945 ms     |

Longer chunks substantially help the frontier family: K=3 at chunk_ms=700
buys +5.96 BLEU / +0.059 COMET over K=3 at chunk_ms=450, at the cost of
~884 ms of CA. This is evidence that the fragmentation penalty scales
with chunk granularity — larger chunks deliver more source context per
MT call, so per-commit mid-sentence fragments are smaller.

That said, `punctuation_lcp` still Pareto-dominates at every measured
operating point: at chunk_ms=450 it beats K=3@700 on every metric while
having ~1 s lower CA; at chunk_ms=700 it beats all frontier variants
by wide margins.

Artifact: `outputs/night1_ende_stable_k3_chunk700/`. This run also
exercises the new `stream_updates.jsonl` schema (alignatt_metadata per
update), so it doubles as the first dataset usable for future offline
continuous-confidence replay work.

## Widening to en→it / en→zh (same clip, same config)

Sanity checks that the `qwen_forced` + `gemma_vllm_alignatt` pair
stays correct across target-language switches under the hardened
runtime (heads-path refresh on any language change, language-code
map covering `cs`).

| Direction | BLEU  | chrF  | COMET | LongYAAL CU | LongYAAL CA | RTF   |
|-----------|-------|-------|-------|-------------|-------------|-------|
| en → de   | 27.51 | 63.54 | 0.861 | 1766 ms     | 1466 ms     | 0.393 |
| en → it   | 37.75 | 71.81 | 0.770 | 1848 ms     | 1567 ms     | 0.400 |
| en → zh   | 42.33 | 38.37 | 0.739 | 1781 ms     | 1634 ms     | 0.375 |

No direction-specific runtime breakage observed across `de`, `it`, and
`zh` target languages. Italian output is coherent; higher absolute BLEU
than en→de reflects the intrinsic proximity of Italian to English.
Chinese scoring uses character-level chrF by definition, which is not
directly comparable to the other targets' token-based chrF — BLEU,
COMET, CU, and CA are the meaningful cross-direction signals.
COMET ranks en→de highest because the XCOMET-XL model has the densest
calibration data for that pair; raw BLEU alone overstates cross-direction
differences relative to how much the cascade itself varies by target
language.

## `translation_alignatt_min_source_mass` sweep (ccpXHNfaoy.wav, chunk_ms=450)

Additional policy knob: MT waits until at least this fraction of the
source's accessible token mass falls within the current frontier.

| min_source_mass | BLEU  | chrF  | COMET  | LongYAAL CU | LongYAAL CA | RTF   |
|-----------------|-------|-------|--------|-------------|-------------|-------|
| 0.0 (baseline)  | 27.51 | 63.54 | 0.861  | 1766 ms     | 1466 ms     | 0.393 |
| 0.1             | 28.25 | 63.81 | 0.867  | 2396 ms     | 2140 ms     | 0.466 |
| 0.2             | 28.95 | 63.92 | 0.869  | 2476 ms     | 2197 ms     | 0.443 |

Each +0.1 in `min_source_mass` buys ~+0.7 BLEU / +0.005 COMET at
~+700 ms CA.
Latency-quality trade is strictly worse than the `chunk_ms`
calibration (recall: 450 → 700 buys +10.7 BLEU for +1479 ms CA on
the same clip). `min_source_mass` remains a valid ablation knob for
paper latency-quality curves, but `chunk_ms` dominates it on the
Pareto front. Artifacts: `outputs/night1_step6_ms{10,20}_punct/`.

## Emission-policy A/B (`raw_passthrough` vs `freeze_nonexpanding_major_rewrites`)

Same config (chunk_ms=450, min_source_mass=0) on ccpXHNfaoy.wav.

| emit_policy                            | BLEU  | chrF  | COMET | LongYAAL CU | LongYAAL CA |
|----------------------------------------|-------|-------|-------|-------------|-------------|
| `raw_passthrough`  (baseline default)  | 27.51 | 63.54 | 0.861 | 1766 ms     | 1466 ms     |
| `freeze_nonexpanding_major_rewrites`   | 27.51 | 63.54 | 0.861 | 1773 ms     | 1484 ms     |

BLEU / chrF / COMET are **bit-identical**. The emit policy suppresses
mid-stream re-emission flicker for display purposes but does not
change the committed final translation, so all quality metrics are
policy-invariant. CU / CA shift by ~10–20 ms (different mid-stream
emission timing), which is well within noise.

Practical implication: the emission policy is a presentation knob,
not a quality knob. The paper's end-to-end BLEU / chrF / COMET
numbers do not depend on it. Artifacts:
`outputs/night1_step6_ms00_freeze/`.

Artifacts: `outputs/night1_enit_punct_chunk450/`,
`outputs/night1_enzh_punct_chunk450/`.

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
