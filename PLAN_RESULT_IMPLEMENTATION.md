# PLAN.md Implementation Results — Gemma-Only Source Aligner

Single-clip deep dive + cross-clip generalization + streaming stability on
`tmp/alignatt_smoke18.wav` (18 s) and a disjoint 18 s slice of
`tmp/ccpXHNfaoy_first75.wav` (seconds 30–48). Branch `maybe_gemma_aligner`.

## Summary

Gemma-4 E4B's self-attention to its `audio_token_id = 258881` span is a
usable source-alignment signal. With top-8 heads from Layer 23, median
filtering, and a single calibrated `word_end_offset_s = 0.480`:

- **Offline forced-alignment MAE: 177 ms** (median 160, P90 400) on the
  calibration clip, **183 ms** (median 160, P90 360) on a cross-content
  18 s slice — within 6 ms of calibration.
- **Streaming drift stdev: 130 ms** vs Qwen baseline 45 ms (~2.9×
  worse); **time-to-stable: 4.5 s** vs Qwen 2.1 s (~2.15× slower).

E4B's *free-run ASR* on conference-speech clips hallucinates — so the
defensible path is **hybrid**: Qwen3-ASR produces the text, Gemma
attention produces the timings. That removes the Qwen3-ForcedAligner-0.6B
dependency while keeping the transcription quality of Qwen3-ASR-1.7B.

Reference landmarks: Qwen3-ForcedAligner ≈ 40 ms MAE, WhisperX ≈ 92 ms,
NFA ≈ 107 ms. Our 177–183 ms is in the WhisperX/NFA regime, ~4.5×
behind the specialized aligner.

## What Was Built

### Alignment backend abstraction (Phase 1)

- `alignment_backend.py` — `AlignmentBackend` ABC + `AlignmentResult`/
  `WordAlignment` dataclasses. Field names (`start_time`, `end_time` in
  seconds) are duck-compatible with `Qwen3ForcedAligner.TimeStamp` so
  `cascade_source_frontier.normalize_word_timestamps_ms` works unchanged.
- `qwen_alignment_backend.py` — baseline wrapping
  `Qwen3ASRModel.transcribe(..., return_time_stamps=True)`.
- `gemma_alignment_probe.py` — `GemmaAttentionAlignmentBackend` with:
  - `transcribe_and_align` (free-run ASR + attention alignment)
  - `align_transcript` (teacher-forced forced alignment — the path that
    works today)
  - `calibrate_alignment_heads{,_forced}` (rank all 42 × 8 heads)
- `hybrid_alignment_backend.py` — `HybridQwenAsrGemmaAlignerBackend` =
  Qwen3-ASR text + Gemma attention timings. Fallback to the ASR
  backend's own timings on any Gemma error.

`qwen3asr_gemma_cascade_core.build_alignment_backend` dispatches on
`config.alignment_backend_name ∈ {"qwen", "gemma_attention",
"hybrid_qwen_asr_gemma_aligner"}`. `transcribe_audio()` delegates
through the backend; `find_end_time` retyped for `WordAlignment`.
`"qwen"` remains the default.

### Harnesses (Phase 2 / 4 / 5 / 7 / 9)

- `run_alignment_single_audio.py`:
  - `baseline` — Qwen teacher bundle on one WAV
  - `gemma_inspect` — Gemma free-run ASR + attention alignment
  - `gemma_forced_align` — teacher-forced alignment from a teacher bundle
  - `gemma_calibrate_heads{,_forced}` — rank all (layer, head) pairs and
    fit the systematic offset against a teacher, persist top-K into an
    `audio_alignment_heads_*.json` bundle
  - `compare` — word-end MAE / median / P90 between two bundles
- `run_streaming_stability.py` — Phase 7 harness: per-tick word drift,
  time-to-stable, identity-change count for any backend.

### Pure-Python invariants (`test_alignment_helpers.py`)

Seven unit tests covering: audio-span detection, 40 ms calibration,
monotone projection, word-span stripping, token→word aggregation,
monotonicity score. All green.

### Environment fix

Pre-existing snapshot-path typo (`/home/.cache/...` with no user dir) in
`qwen3asr_gemma_cascade_core.py` now resolves via `_resolve_hf_snapshot`
which falls back to `~/.cache/...`.

## Empirical Results

### Phase 4/5 — attention-based forced alignment + head selection

Calibration on `smoke18` using Qwen teacher bundle:

| Heads | Monotonicity | Coverage | MAE (no offset) |
|---|---:|---:|---:|
| top-8 L23 H0-H7 | 0.816 – 0.898 | 1.00 | **310–730 ms** |
| Best single head (L23 H2) | 0.816 | 1.00 | **310 ms** |
| Worst scored head | ~0.3 | 1.00 | > 10 s |

All 8 heads of **Layer 23** rank in the top-10 by MAE — a clean
single-layer alignment-head cluster, consistent with middle-layer
findings in the token-alignment-heads literature.

### Systematic offset calibration

Per-word signed error has median **+480 ms** (Gemma peaks after the
acoustic word boundary because a causal LLM must see a bit past the
word before committing). Subtracting that single scalar per
`(language, model)` cuts MAE from 525 ms → **177 ms** on the
calibration clip.

Stored in the heads JSON under `word_end_offset_seconds` — downstream
consumers load the calibrated bundle without re-fitting.

### Phase 2/9 — offline accuracy

Top-8 L23 + 480 ms offset, top-k=8:

| Clip | Words | MAE | Median | P90 |
|---|---:|---:|---:|---:|
| `smoke18` (calibration) | 35 | **177 ms** | 160 ms | 400 ms |
| `ccp30s_48s` (different content) | 36 | **183 ms** | 160 ms | 360 ms |

Second clip has truly different content: "Cake, and show that large
language models can effectively decompose goals into steps. However,
previous work mainly focuses on planning for the abstract goals..." —
same speaker, different section of the same talk. Heads and offset
generalize within noise.

### Phase 7 — streaming stability

`ccpXHNfaoy 5–25 s`, tick every 2 s, 11 ticks total:

| Metric | Qwen baseline | Hybrid | Ratio |
|---|---:|---:|---:|
| mean stdev of word-end | 45 ms | **130 ms** | 2.9× |
| median stdev | 0 ms | 76 ms | — |
| mean drift range | 112 ms | 299 ms | 2.7× |
| max drift range | 800 ms | 1240 ms | 1.55× |
| num backward jumps | 12 | 20 | 1.67× |
| max backward jump | 720 ms | 760 ms | 1.06× |
| time-to-stable (mean) | 2.1 s | **4.5 s** | 2.15× |
| identity changes | 8 | 8 | = (same ASR text) |

**Interpretation**: the Gemma aligner adds ~2.4 s to the
utterance-commit latency in the cascade, with 3× noisier per-word
timing but comparable worst-case backward-jump magnitude. Not a
pathological failure mode.

### Phase 3 — free-run Gemma ASR investigation (negative result)

Gemma E4B free-run ASR on `smoke18` **hallucinates**:

| Prompt variant | Output (audio is "Hi, I'm Siyu Yuan from Fudan University...") |
|---|---|
| Text-before-audio, `"in English into text"` | "Hi, sir. For the university, I need to do the work. This is not for the university. In the name of Professor Sanchez..." |
| Text-before-audio, `"in English into English text"` | "Hi sir, for the university, I need to do the work..." (basically the same) |
| **Audio-before-text, `"in its original language"` (cookbook exact)** | "Hi, I'm a student from University of Nottingham and I'm interning at Global Logistics for Sustainable Planning..." |

**Diagnosis**: E4B recognizes the acoustic shape of an introduction
("I'm X from Y University") but fabricates the content. Google's claimed
FLEURS WER 0.08 is on clean read-aloud native-accent single-speaker
audio; a non-native speaker at a conference (Chinese accent saying
"Fudan", "Siyu Yuan") is well outside the FLEURS distribution.

The `AutoModelForMultimodalLM` / `AutoModelForImageTextToText` auto
classes **both resolve to the same** `Gemma4ForConditionalGeneration`, so
the class swap alone is cosmetic — the audio-before-text reordering is
the only substantive change. It produces a *different* hallucination,
not a *correct* transcription.

Monotonicity of the Gemma attention signal also drops from 0.98 (text
first) to 0.76 (audio first) in free-run mode, suggesting the text-first
template keeps attention more sharply peaked on the audio even though
the emitted text is still wrong. This is an extra reason to stay on the
hybrid path.

## Cookbook-Alignment Changes (post-initial-results)

After the user pointed out the recommended Gemma cookbook pattern, the
probe was updated to match exactly:

- `AutoModelForMultimodalLM` (cookbook) instead of
  `AutoModelForImageTextToText` (same underlying class).
- `{"type": "audio", ...}` block placed **before** `{"type": "text", ...}`
  in the user message.
- Text prompt changed to the cookbook's `"in its original language"`
  variant.
- Dtype cast simplified to a single `.to(model.device)`; per-key float
  cast to `model.dtype` removed (letting the model auto-cast internally
  avoids a bf16 quantization pass over the mel features).

These changes do not fix the free-run hallucination (see above). The
calibrated `audio_alignment_heads_google_gemma-4-E4B-it_en_forced.json`
bundle was fit under the *text-first* message ordering. Forced
alignment under the new *audio-first* ordering may need heads + offset
re-calibration to hold the 177 ms MAE number — this is the first open
item below.

## What the Cascade Can Do Today

```python
from qwen3asr_gemma_cascade_core import config, run_stream
config.alignment_backend_name = "hybrid_qwen_asr_gemma_aligner"
run_stream("test-set/audio/ccpXHNfaoy.wav", chunk_ms=960)
```

The cascade transparently uses Qwen3-ASR for transcription and Gemma
attention for word timestamps. If Gemma alignment fails for any reason
(short utterance, audio cap, head mismatch), the backend falls back to
Qwen3-ForcedAligner's own timings for that tick. Default
`alignment_backend_name = "qwen"` preserves the baseline bit-for-bit.

## Outstanding Work

1. **Re-calibrate heads + offset under the new audio-first ordering.**
   The 177 ms MAE was measured with text-first. Expected to stay similar
   but must be re-verified.
2. **Full cascade run** with `hybrid_qwen_asr_gemma_aligner` vs Qwen
   baseline: BLEU / chrF / LongYAAL-CU / LongYAAL-CA on one talk
   (`ccpXHNfaoy.wav`). Streaming numbers predict ~+500 ms CU; need the
   actual delta.
3. **Multi-clip head calibration** (3–5 clips across speakers/accents)
   to check whether L23 + 480 ms is a single-speaker local optimum or a
   robust choice.
4. **Enforce the 30 s audio cap** (`audio_seq_length = 750` × 40 ms) in
   `align_transcript`. Longer utterances currently truncate silently.
5. **E4B ASR investigation** (optional): try E2B, or a different
   checkpoint such as `principled-intelligence/gemma-4-E4B-it-text-only`,
   or prompt strategies forcing transcription-only mode. If E4B simply
   can't transcribe conference-quality audio, accept the hybrid path as
   the final architecture and document the trade-off honestly.

## Artifacts

Under `tmp/alignment_research/`:

- `smoke18_qwen_teacher.json` — Qwen baseline (18 s calibration clip)
- `smoke18_gemma_forced_top8.json` — no-offset Gemma forced alignment
- `smoke18_gemma_forced_calibrated.json` — with +480 ms offset
- `smoke18_gemma_audio_first.json` — free-run under new prompt ordering
- `ccpXHNfaoy_18s_*.json` — same-content cross-clip check
- `ccpXHNfaoy_30s_48s_*.json` — different-content generalization
- `ccp25_qwen_ticks.json`, `ccp25_hybrid_ticks.json` — streaming
  stability tick series + metrics

Under `assets/attention_heads/`:

- `audio_alignment_heads_google_gemma-4-E4B-it_en_forced.json` — top-16
  calibrated heads + 480 ms offset
- `audio_alignment_heads_google_gemma-4-E4B-it_en_forced.full_ranking.json`
  — all 336 scored heads for ablation

## Files Added / Modified

- **Added**: `alignment_backend.py`, `qwen_alignment_backend.py`,
  `gemma_alignment_probe.py`, `hybrid_alignment_backend.py`,
  `run_alignment_single_audio.py`, `run_streaming_stability.py`,
  `test_alignment_helpers.py`.
- **Modified**: `qwen3asr_gemma_cascade_core.py` (backend dispatch +
  snapshot path fix).

Seven pure-Python invariants in `test_alignment_helpers.py` lock the
non-trivial data-flow rules; all green.
