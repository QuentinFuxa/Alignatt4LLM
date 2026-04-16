# PLAN.md Audit Note — Structural Fixes Pass

Single iteration scope: code-level integrity fixes from PLAN.md
"Critical Findings" + "Immediate First Steps". GPU-requiring measurements
(re-calibration, robustness sweep, cascade run) are scoped to a separate
runtime pass.

## What Changed

### 1. Runtime default head bundle now points to the calibrated `_forced` file
`qwen3asr_gemma_cascade_core.py:244-251` — default
`gemma_audio_alignment_heads_path` is now
`audio_alignment_heads_google_gemma-4-E4B-it_en_forced.json` (the bundle
that produced the 177 ms MAE), and `build_alignment_backend` raises
`FileNotFoundError` when the configured path does not exist. The plain
`en.json` bundle is no longer wired into `gemma_attention` or `hybrid_*`
silently.

### 2. Hybrid backend exposes Gemma-vs-fallback usage explicitly
`hybrid_alignment_backend.py` — every `transcribe_and_align` return now
populates `gemma_alignment_used: bool`, `fallback_reason: str | None`,
`gemma_error: str | None`, and word-count truncation flags. Gemma
exceptions are caught and reported as `gemma_exception` instead of
crashing the cascade tick. Hybrid metrics can now be split into pure
Gemma ticks vs Qwen-fallback ticks.

### 3. Long-audio guard is loud, not silent
`gemma_alignment_probe.py` — new `GemmaAudioTooLongError` and
`_enforce_audio_cap` (default 30 s, configurable via
`max_audio_seconds`) called from both `transcribe_and_align` and
`align_transcript`. Audio past the encoder cap now raises with the
duration in the error message. Eliminates the silent-truncation risk
the implementation notes flagged.

### 4. Streaming harness records and aggregates fallback usage
`run_streaming_stability.py` — `StreamingTick` carries per-tick
`diagnostics`; `compute_streaming_metrics` aggregates
`fallback_aware_ticks`, `gemma_used_ticks`, `fallback_ticks`,
`fallback_rate`, and a `fallback_reasons` histogram. Per-tick prints
also show `gemma_used=...`. Hybrid stability numbers can no longer hide
fallback behind a single mean.

### 5. Fair Gemma ASR benchmark harness
`run_gemma_asr_fairness.py` (new) — controlled matrix over
{decode_path: generate vs manual_greedy} × {prompt_order: audio_first
vs text_first} × {prompt_wording: cookbook vs explicit_english} ×
{input_cast: to_device vs to_device_and_dtype}. Default mode runs five
key cells (decode-path equivalence + each single-axis ablation);
`--full-matrix` runs all 16. Outputs JSON with every variant's exact
transcript + WER + CER vs a trusted reference, plus a decode-path
agreement check at T=0. No alignment heads loaded — pure ASR fairness.

## Gemma ASR Fairness Benchmark — Verdict

Ran `run_gemma_asr_fairness.py` on `tmp/alignatt_smoke18.wav` (18 s, Siyu
Yuan / Fudan introduction). Reference: Qwen3-ASR transcript. Results:

| decode | order | wording | cast | WER | CER |
|---|---|---|---|---:|---:|
| generate | audio_first | cookbook_original_language | to_device | 0.829 | 0.672 |
| manual_greedy | audio_first | cookbook_original_language | to_device | 0.829 | 0.672 |
| generate | text_first | cookbook_original_language | to_device | 0.943 | 0.667 |
| generate | audio_first | explicit_english | to_device | 0.886 | 0.728 |
| generate | audio_first | cookbook_original_language | to_device_and_dtype | 0.829 | 0.672 |

Decode-path equivalence under matched prompt+cast at T=0: **generate ≡
manual_greedy** ✓. The cascade probe's manual greedy loop is not
introducing the bad transcription.

Sample outputs (audio is "Hi, I'm Siyu Yuan from Fudan University..."):
- audio_first / cookbook: "Hi, I'm a student from University of
  Nottingham and I'm interning at Global Logistics for Sustainable
  Planning. I'm really interested in how..."
- text_first / cookbook: "Hi, sir. For the university, I need to do the
  work. This is not for the university. In the name of Professor
  Sanchez, for the language learn..."
- audio_first / explicit_english: "Hi, I'm a student from University of
  Nottingham and I'm interested in learning about sustainable farming
  practices in India..."

**Verdict — SUPERSEDED.** This benchmark was run with
`attn_implementation="eager"`, which is the dominant cause of the bad
ASR (see ITERATION_RESULT.md §Root Cause). With default attention,
Gemma achieves WER 0.03–0.26 on the same clips. The conclusion below
that Gemma ASR is "unfit" no longer holds.

## Recommendation

**Hybrid (Option B from PLAN.md Phase 7)**: Qwen3-ASR for transcription,
Gemma attention for word timings. Justification:

1. **SUPERSEDED**: The ≥0.83 WER was caused by eager attention, not by
   the model's ASR capability. With default attention, WER is 0.03–0.26.
2. The attention-based aligner produces 177–183 ms MAE word-end timings
   on the same clip family — within the WhisperX/NFA regime — and now
   has explicit fallback accounting in the hybrid path so its
   contribution is auditable per tick.
3. The forced-bundle default + loud-failure on missing heads removes
   the silent-degradation footgun that prompted this audit.

Runtime ↔ calibration mismatch is now closed: forced-alignment prompting
was reverted to text-first to match `_forced.json`, and revalidation
reproduces **187 ms MAE / 200 ms median / 400 ms P90** — within 10 ms of
the previously reported number. The hybrid architecture can be quoted
against the measured alignment quality.

## Aligner Revalidation Under Matched Prompt Contract

Measured on smoke18 (35 words) against the Qwen teacher bundle:

| Prompt ordering in `_prepare_forced_alignment_inputs` | Heads bundle | MAE | Median | P90 |
|---|---|---:|---:|---:|
| audio-first (cookbook) + audio-first heads | `_forced_audiofirst.json` (offset 0.64s) | **502 ms** | 200 ms | 1240 ms |
| text-first + text-first heads | `_forced.json` (offset 0.48s) | **187 ms** | 200 ms | 400 ms |

Gap: 2.7× on MAE, 3.1× on P90. Matches the monotonicity drop the
implementation notes reported (0.98 → 0.76). Under the cookbook
audio-first prompting, the assistant-token attention into the audio
span is materially less peaked.

### Applied fix

`gemma_alignment_probe.py _prepare_forced_alignment_inputs` now puts
the text block **before** the audio block, decoupling the forced-
alignment prompting from the free-run ASR prompting:

- `_render_asr_messages` (used by `transcribe_and_align`): still
  cookbook audio-first (Google's recommended contract for ASR output;
  the cascade doesn't read Gemma's free-run output anyway).
- `_prepare_forced_alignment_inputs` (used by `align_transcript`,
  `calibrate_alignment_heads_forced`, and therefore the hybrid path):
  text-first, which produces the sharp attention signal the aligner
  depends on.

Default `config.gemma_audio_alignment_heads_path` is left at
`_forced.json` (the text-first calibration), which now matches the
runtime ordering again — closing the PLAN §Critical Findings #1
integrity issue. The `_forced_audiofirst.json` file is kept for
ablation; it is not the default.

This split is defensible: free-run ASR and teacher-forced alignment are
two different tasks; the cookbook ordering constraint comes from
Google's ASR fine-tuning, not from any property of the attention heads.
The text-first choice is backed by a measured 2.7× MAE improvement on
the same clip, not by cherry-picking.

## What Is Still Open

Remaining items that need a runtime pass — each is now one command:

- **Hybrid fallback audit** on one talk with the new diagnostics:
  `run_streaming_stability.py --wav ... --tag ... --hybrid --heads-path
  assets/attention_heads/audio_alignment_heads_google_gemma-4-E4B-it_en_forced.json`.
  The `fallback_rate` / `fallback_reasons` fields now surface the
  per-tick split directly.
- **Small robustness set** (3–5 clips, diverse speakers/accents) reusing
  `_forced.json` — confirms whether L23 + 0.48 s generalizes or is
  smoke18-local.
- **Cascade-level comparison** (BLEU / chrF / LongYAAL-CU/CA) of
  `hybrid_qwen_asr_gemma_aligner` vs `qwen` baseline on one talk.

## Final Recommendation Posture (unchanged pending the runtime pass)

The structural fixes above make the existing claims auditable; they
don't change the verdict. The defensible path remains **hybrid**
(Qwen3-ASR text + Gemma attention timings) until either (a) the ASR
fairness harness shows competitive Gemma transcription, or (b)
re-calibration shows the alignment signal collapses under the cookbook
prompt. Both are now one command away.

## Files Modified / Added

- Modified: `qwen3asr_gemma_cascade_core.py` (default head bundle +
  strict resolution), `hybrid_alignment_backend.py` (fallback
  diagnostics), `gemma_alignment_probe.py` (long-audio guard; split
  prompt contracts — cookbook for free-run ASR, text-first for forced
  alignment), `run_streaming_stability.py` (per-tick fallback accounting),
  `test_alignment_helpers.py` (guard + fallback invariants).
- Added: `run_gemma_asr_fairness.py` (controlled ASR matrix),
  `PLAN_AUDIT_NOTE.md` (this note),
  `assets/attention_heads/audio_alignment_heads_google_gemma-4-E4B-it_en_forced_audiofirst.json`
  (ablation bundle).
- Artifacts: `tmp/alignment_research/gemma_asr_fairness_smoke18.json`,
  `smoke18_gemma_forced_audiofirst.json`,
  `smoke18_gemma_forced_textfirst_revalidate.json`.
