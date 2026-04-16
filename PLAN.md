# PLAN

Read `CLAUDE.md` and `AGENTS.md` first.


## Purpose

This file is the handoff plan for the next agent.

We are now in delivery mode, not open-ended exploration mode.

The target is a deliverable, reproducible, paper-defensible implementation of
the Qwen3-ASR + Gemma AlignAtt cascade inside the real **SimulStream**
framework, with one validated operating point per language and per latency
regime.


## Mandatory Reading

Before editing code or launching any expensive run, read these files in order:

1. `CLAUDE.md`
2. `AGENTS.md`
3. `assets/alignatt_doc/ALIGNATT_LLM.md`
4. `assets/alignatt_doc/E4B_ALIGNATT_CASCADE_DESIGN.md`
5. `assets/alignatt_doc/alignatt_markdown.md`
6. `assets/alignatt_doc/alignatt_whipser.py`

Then read the core implementation:

1. `qwen3asr_gemma_cascade_core.py`
2. `cascade_mt_backend.py`
3. `cascade_emission.py`
4. `cascade_text_surface.py`
5. `evaluate_cascade_outputs.py`

Then read the new SimulStream-facing files:

1. `cascade_simulstream_processor.py`
2. `run_simulstream_evaluation.py`
3. `run_simulstream_batch.py`
4. `benchmark_simulstream_speed.py`

Then read the SimulStream framework contract from the installed package:

1. `.venv-inference/lib/python3.13/site-packages/simulstream/server/speech_processors/__init__.py`
2. `.venv-inference/lib/python3.13/site-packages/simulstream/server/speech_processors/base.py`
3. `.venv-inference/lib/python3.13/site-packages/simulstream/server/speech_processors/incremental_output.py`
4. `.venv-inference/lib/python3.13/site-packages/simulstream/server/message_processor.py`
5. `.venv-inference/lib/python3.13/site-packages/simulstream/server/websocket_server.py`

Useful result bundles to inspect before rerunning anything:

1. `outputs/phase0_v4_ende_reproduce`
2. `outputs/phase1_v2_enit_validate`
3. `outputs/phase1_v1_enzh_validate_reemit`
4. `outputs/phase4_v4_enit_shared_kernel`
5. `outputs/phase5_v1_ende_minmass20`
6. `outputs/compute_unaware_chunk800_20260415T154922Z`


## Non-Negotiable Constraints

- The final delivery must use **SimulStream**, not only the local research
  harness.
- Use `.venv-inference`.
- Model reload is expensive. Reuse hot models whenever possible.
- Do not run broad sweeps until the current objective is already validated on a
  single audio.
- Do not add ad hoc lexical fixes, dataset-specific rewrites, or benchmark
  patches.
- Prefer the cleanest architecture that we could defend in a paper.
- If something is only "working" because of module-global state or accidental
  ordering, treat that as broken.


## Delivery Targets

We need one delivery-ready operating point for each of these six slots:

- `en->de` under `< 2 s LongYAAL Compute-Unaware`
- `en->it` under `< 2 s LongYAAL Compute-Unaware`
- `en->zh` under `< 2 s LongYAAL Compute-Unaware`
- `en->de` under `< 4 s LongYAAL Compute-Unaware`
- `en->it` under `< 4 s LongYAAL Compute-Unaware`
- `en->zh` under `< 4 s LongYAAL Compute-Unaware`

The chosen operating points must satisfy two kinds of constraints:

- quality / latency metrics on the full evaluation set
- actual runtime credibility through the real SimulStream path

Do not accept a config only because the corpus average looks good. Track
per-audio violations as well.


## Current Conceptual Understanding

The core AlignAtt idea is still:

- generate a target draft
- inspect attention-based source provenance for each target token
- stop before the first token that depends on inaccessible source context
- emit only the accepted prefix

In Whisper, this was naturally tied to decoder cross-attention over encoder
frames.

In this Gemma-based LLM implementation, the same logic is approximated through:

- a source-aware prompt layout
- selected self-attention heads
- `qk_fast` or `eager` probing
- a prefix-online scan over the draft
- acceptance outside the model

The strongest recent mechanism is the provenance-aware acceptance criterion via
`translation_alignatt_min_source_mass`, which rejects a token when the source
argmax looks acceptable but the accessible-source attention mass is too weak.


## What Is Already Good

- The overall LLM AlignAtt decomposition is sound:
  - fast draft
  - separate probe
  - external acceptance
- The accepted-prefix semantics are much cleaner than a naive Whisper port.
- Multilingual support already exists for `de`, `it`, and `zh`.
- Shared-kernel heads look promising and likely delivery-friendly.
- Provenance-aware acceptance is a real mechanism improvement, not just harness
  cleanup.
- The new provenance serialization fix in `qwen3asr_gemma_cascade_core.py` is
  good and should be preserved.


## What The Last Agent Added

The new work is real and useful:

- `cascade_simulstream_processor.py`
  - first real `SpeechProcessor` wrapper around the cascade
- `run_simulstream_evaluation.py`
  - produces evaluation-compatible artifacts through the SimulStream path
- `run_simulstream_batch.py`
  - keeps models hot across multiple audios
- `benchmark_simulstream_speed.py`
  - measures wallclock, RTF, chunk times, peak GPU memory
- `qwen3asr_gemma_cascade_core.py`
  - now persists `translation_alignatt_min_source_mass`
  - now enriches `run_provenance` with git SHA / framework mode / languages

Lightweight verification already done:

- `python -m py_compile cascade_simulstream_processor.py benchmark_simulstream_speed.py run_simulstream_batch.py run_simulstream_evaluation.py qwen3asr_gemma_cascade_core.py`
  - passes


## Critical Review Of The New SimulStream Work

The new integration is a strong step forward, but it is **not yet delivery-safe**.

### Blocker 1 - Session isolation is wrong

`cascade_simulstream_processor.py` uses a class-level shared `_core`, while
`qwen3asr_gemma_cascade_core.py` stores mutable stream state in module globals:

- `config`
- `state`
- `translation_units`
- `mt_backend`

This conflicts with the SimulStream execution model, because the framework
creates a pool of processor instances and expects them to be independently
usable.

Implication:

- current implementation is not trustworthy for pooled / multi-client /
  multi-session serving
- even if it "works" for a single local run, it is architecturally unsafe

### Blocker 2 - Reconfiguration after first load is stale

The current processor mutates `core.config` in `__init__()` and in
`set_source_language()` / `set_target_language()`, but the MT backend is only
built once and AlignAtt heads are only loaded once.

This means the following may silently fail to update after the first load:

- language-specific heads
- `translation_alignatt_top_k_heads`
- `translation_alignatt_probe_mode`
- any other backend-initialized probe setting

Implication:

- a process that first loads `en->de` may continue using stale German heads for
  later `en->it` / `en->zh`
- current SimulStream results should be treated as provisional until this is fixed

### Blocker 3 - EOS is missing from `stream_updates`

`run_simulstream_evaluation.py` and `run_simulstream_batch.py` call
`end_of_stream()`, but they do not append a final incremental update to
`stream_updates.jsonl` when EOS emits something new.

Implication:

- the serialized stream trace can be incomplete
- replay/debugging and latency analysis can be inconsistent with the final text

### Blocker 4 - Batch updates have no audio identity

`run_simulstream_batch.py` concatenates updates from multiple audios into one
`stream_updates.jsonl`, but each update record lacks a `wav_name` or `audio_id`.

Implication:

- batch stream traces are ambiguous
- later replay or debugging is harder than necessary

### Blocker 5 - Benchmark GPU peak memory is not cleanly measured

`benchmark_simulstream_speed.py` reports `torch.cuda.max_memory_allocated()`
without resetting the peak before the benchmarked section.

Implication:

- the reported number mixes model load and run-time memory
- speed artifacts are less useful for deployment decisions

### Minor hygiene note

- `.claude/ralph-loop.local.md` is a local artifact and should not be committed.


## Important Consequence

The current SimulStream outputs and measurements are useful as **signals**, but
not yet as final delivery truth.

Until Blockers 1 and 2 are fixed, do **not** present the current SimulStream
results as fully validated production numbers.


## Recommended Design Direction

Do not paper over the current issues with one more layer of mutable global
patching.

The clean direction is:

- one shared loaded model bundle
  - ASR weights
  - Gemma weights
  - tokenizer
- one per-session mutable cascade state
  - source audio buffer
  - ASR incremental state
  - accepted / draft translation state
  - emission surface state
- explicit backend refresh when language-dependent AlignAtt artifacts change

The minimal acceptable fallback, if time pressure is extreme, is:

- document `pool_size=1`
- make the processor explicitly single-session / single-language per process
- still fix stale-head reload and EOS tracing

But the preferred direction is real state isolation.


## Current Best Reference Points

These are reference points from earlier work and should be used to guide the
next experiments after the SimulStream blockers are fixed.

### `< 2 s` family

Control-audio reference at `chunk_ms = 450`:

- `en->de`
  - `outputs/phase0_v4_ende_reproduce`
  - `BLEU 28.22`, `chrF 63.53`, `LongYAAL CU 1747.19`
- `en->it`
  - `outputs/phase1_v2_enit_validate`
  - `BLEU 36.87`, `chrF 71.48`, `LongYAAL CU 1813.70`
- `en->zh`
  - `outputs/phase1_v1_enzh_validate_reemit`
  - `BLEU 41.85`, `chrF 38.32`, `LongYAAL CU 1762.95`

Shared-kernel references are strong for `it` and `zh`, and likely still worth
testing for `de`.

For `en->de`, provenance-aware acceptance on control audio gives:

- `outputs/phase5_v1_ende_minmass10`
  - `BLEU 28.14`, `chrF 63.55`, `CU 1790.72`
- `outputs/phase5_v1_ende_minmass20`
  - `BLEU 29.58`, `chrF 64.00`, `CU 1989.86`
- `outputs/phase5_v1_ende_minmass30`
  - `BLEU 29.34`, `chrF 64.09`, `CU 2162.88`

Interpretation:

- `min_source_mass = 0.2` is a promising quality point on control audio
- but it may become too expensive or too fragile on a broader audio set
- do not assume it is the final `<2s` winner without fresh SimulStream validation

### `< 4 s` family

The main high-quality reference family is:

- `outputs/compute_unaware_chunk800_20260415T154922Z`
  - `BLEU 38.76`, `chrF 68.09`, `LongYAAL CU 3716.85`

This remains the natural starting point for the `<4s` regime.


## Candidate Matrix To Start From

Do not search a huge grid.

Start from this small matrix **after** fixing the SimulStream blockers.

### `< 2 s`

Base family:

- `chunk_ms = 450`
- `min_start_seconds = 2.0`
- `max_history_utterances = 1`
- `partial_max_new_tokens = 16`
- `partial_followup_max_new_tokens = 8`
- `translation_alignatt_inaccessible_ms = 0`
- `translation_alignatt_rewind_threshold = 8`

Initial candidates:

- `en->de`
  - baseline heads
  - baseline heads + `translation_alignatt_min_source_mass = 0.2`
  - shared-kernel heads + `translation_alignatt_min_source_mass = 0.2`
- `en->it`
  - shared-kernel heads
- `en->zh`
  - shared-kernel heads

### `< 4 s`

Base family:

- start from `chunk_ms = 800`
- start from the quality-favoring compute-unaware family

Initial candidates:

- `en->de`
  - current `chunk800` quality reference
- `en->it`
  - same family with correct `it` heads
- `en->zh`
  - same family with correct `zh` heads or shared-kernel heads

Only tune beyond this if the initial point fails the target or the speed gate.


## Execution Order

### Phase 0 - Re-read And Orient

1. Read the mandatory documents above.
2. Confirm the current git state.
3. Do not trust the old `PLAN.md` assumptions that said the SimulStream path was
   fully validated.

### Phase 1 - Fix Correctness Before New Expensive Runs [DONE]

1. [DONE] Fix session isolation or explicitly enforce/document single-processor
   behavior.
   → Added `_active_instance_id` guard to `CascadeAlignAttProcessor`; documented
     `pool_size=1` constraint.
2. [DONE] Fix stale backend / stale head reconfiguration.
   → Added `refresh_alignatt_artifacts()` to `TransformersAlignAttGemmaMTBackend`
     (removes old hooks, reloads heads, re-registers recorders).
   → `set_source_language()` / `set_target_language()` now call it when the heads
     path actually changes.
3. [DONE] Append EOS updates into serialized `stream_updates`.
   → Both `run_simulstream_evaluation.py` and `run_simulstream_batch.py` now
     append a final update with `is_eos: true` when EOS produces new output.
4. [DONE] Add `wav_name` or equivalent identity to batch `stream_updates`.
   → Every stream update in `run_simulstream_batch.py` now carries `wav_name`.
5. [DONE] Reset GPU peak-memory counters before benchmark timing.
   → `benchmark_simulstream_speed.py` calls `torch.cuda.reset_peak_memory_stats()`
     before the timed section.
6. [DONE] Keep `.claude/ralph-loop.local.md` out of commits.
   → Added to `.gitignore`.

### Phase 2 - Cheap Verification [PARTIAL]

Run cheap checks first:

- [DONE] `py_compile` on all touched Python files — passes
- only minimal targeted tests if they protect a real invariant

Good invariants to protect:

- language change really refreshes the heads/backend
- final EOS emission is serialized
- batch updates include an audio identifier

Do not write lots of low-value tests.

### Phase 3 - Single-Audio SimulStream Smoke Validation [DONE]

Control audio `ccpXHNfaoy.wav` through the real SimulStream path:

1. [DONE] `en->de` `<2s` — BLEU 28.22, chrF 63.53, CU 1747.19 (exact match)
2. [DONE] `en->de` `<4s` — BLEU 38.01, chrF 67.17, CU 3339.69 (close to ref)
3. [DONE] `en->it` `<2s` — BLEU 36.87, chrF 71.48, CU 1813.70 (exact match)
4. [DONE] `en->zh` `<2s` — BLEU 41.85, chrF 38.32, CU 1762.95 (exact match)

RTF concern: all configs at ~1.0 RTF on 360s control audio. This is the
speed gate question for Phase 4.

Artifacts in `outputs/phase3_simulstream_*`.

### Phase 4 - Real Speed Gate

Before any full-set run, benchmark each surviving candidate through the real
SimulStream processor on:

- the control audio
- a tiny extra sanity set of short/medium audios per language

Record:

- `BLEU`
- `chrF`
- `LongYAAL CU`
- `LongYAAL CA`
- `RTF`
- mean chunk time
- p95 chunk time
- max chunk time
- EOS time
- peak GPU memory

Operational rule of thumb:

- reject if `RTF >= 1.0`
- strongly prefer `RTF <= 0.6`
- reject if p95 chunk compute time is too close to or above the chunk interval

### Phase 5 - Full-Set Selection

Only after the fixed SimulStream path passes the speed gate:

1. run the small candidate matrix on the full audio set
2. compare all six delivery slots
3. choose one winner per slot

Track both:

- corpus-level metrics
- per-audio regime violations

If a config is good on average but breaks the target on too many audios, it is
not a safe delivery config.


## Practical Commands

Use these as starting points, after the correctness fixes.

Syntax check:

```bash
.venv-inference/bin/python -m py_compile \
  cascade_simulstream_processor.py \
  run_simulstream_evaluation.py \
  run_simulstream_batch.py \
  benchmark_simulstream_speed.py \
  qwen3asr_gemma_cascade_core.py
```

Single-audio SimulStream eval:

```bash
.venv-inference/bin/python run_simulstream_evaluation.py \
  --wav test-set/audio/ccpXHNfaoy.wav \
  --output-dir outputs/tmp_simulstream_check \
  --chunk-ms 450 \
  --source en \
  --target de \
  --min-start-seconds 2.0 \
  --max-history-utterances 1 \
  --partial-max-new-tokens 16 \
  --partial-followup-max-new-tokens 8
```

Speed benchmark:

```bash
.venv-inference/bin/python benchmark_simulstream_speed.py \
  --wav test-set/audio/ccpXHNfaoy.wav \
  --chunk-ms 450 \
  --source en \
  --target de
```

Small batch run:

```bash
.venv-inference/bin/python run_simulstream_batch.py \
  --wavs test-set/audio/myfXyntFYL.wav test-set/audio/ccpXHNfaoy.wav \
  --output-dir outputs/tmp_simulstream_batch \
  --chunk-ms 450 \
  --source en \
  --target de
```


## What Not To Do

- Do not launch full-set runs before fixing the SimulStream blockers.
- Do not trust current SimulStream numbers as final delivery truth yet.
- Do not run large grids.
- Do not restart the hot model stack unless necessary.
- Do not patch isolated examples with heuristics.
- Do not let "delivery urgency" justify architecture we could not defend in a
  paper.


## Definition Of Done

We are handoff-ready only when all of the following are true:

- the SimulStream processor is architecturally correct enough to trust
- per-session behavior is isolated or explicitly constrained and documented
- language/head reconfiguration is deterministic and not stale
- SimulStream artifacts faithfully include final EOS emissions
- speed measurements come from the real SimulStream path
- every result bundle is self-describing
- one operating point is selected for each of the six delivery slots
- those operating points survive full-set evaluation credibly


## First Task For The Next Agent

Do this first:

1. read the mandatory files
2. inspect the current SimulStream implementation
3. fix the state-isolation and stale-head problems before any new broad run

If you do nothing else, do not skip that.
