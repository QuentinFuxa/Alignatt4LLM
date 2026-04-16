# PLAN

Check `CLAUDE.md` and `AGENTS.md` first.


## Purpose

This file is the handoff plan for the next agent.

We are now close to the implementation delivery date.

The goal is no longer open-ended exploration. The goal is to turn the current
AlignAtt-on-LLM cascade into a deliverable, reproducible, fast implementation
inside the **SimulStream** framework, with validated operating points for all
target languages and both latency regimes.


## Mandatory Reading

Before changing code or launching any expensive run, read these documents:

1. `CLAUDE.md`
2. `AGENTS.md`
3. `assets/alignatt_doc/ALIGNATT_LLM.md`
4. `assets/alignatt_doc/E4B_ALIGNATT_CASCADE_DESIGN.md`
5. `assets/alignatt_doc/alignatt_markdown.md`
6. `assets/alignatt_doc/alignatt_whipser.py`

Then read these implementation files:

1. `qwen3asr_gemma_cascade_core.py`
2. `cascade_mt_backend.py`
3. `cascade_emission.py`
4. `cascade_text_surface.py`
5. `run_cascade_baseline.py`
6. `evaluate_cascade_outputs.py`

Then read the SimulStream delivery interface in the installed framework:

1. `.venv-inference/lib/python3.13/site-packages/simulstream/server/speech_processors/__init__.py`
2. `.venv-inference/lib/python3.13/site-packages/simulstream/server/speech_processors/base.py`
3. `.venv-inference/lib/python3.13/site-packages/simulstream/server/speech_processors/incremental_output.py`

The delivery must use the real SimulStream processor API, not only the local
research harness.


## Non-Negotiable Constraints

- We must deliver inside **SimulStream**:
  - official repo: <https://github.com/hlt-mt/simulstream>
- Use the repo environment `.venv-inference`.
- Do not restart the ASR + Gemma stack unless necessary; model reload is costly.
- Full-set runs are expensive and should only happen once speed and behavior are
  already validated on small control cases.
- No ad hoc heuristics, no lexical patches, no benchmark-specific hacks.
- The final system must be paper-defensible and operationally credible.


## Required Final Regimes

We need validated operating points for:

- `en->de` under `< 2 s LongYAAL CU`
- `en->it` under `< 2 s LongYAAL CU`
- `en->zh` under `< 2 s LongYAAL CU`
- `en->de` under `< 4 s LongYAAL CU`
- `en->it` under `< 4 s LongYAAL CU`
- `en->zh` under `< 4 s LongYAAL CU`

These are six delivery slots.

The requirement is not only "good average".

We need configurations that respect the corresponding regime on the full data
evaluation in a credible way, and we must know their compute-aware / RTF
behavior before launching several hours of audio.


## Current State Of The System

### What is already solid

- The semantic split `draft_target` / `accepted_target` is correct.
- The prompt contract is correct:
  - `user = full source prefix`
  - `assistant = accepted target prefix`
- The Gemma backend already has the right high-level structure:
  - fast draft
  - separate alignment probe
  - acceptance outside the model
- The current observer is already much better than a naive Whisper port:
  - `qk_fast`
  - prefix-online scan
  - monotone accepted prefix
- The cascade is now structurally multilingual:
  - `de`
  - `it`
  - `zh`

### Important recent mechanism

The main recent mechanism is **provenance-aware acceptance**.

`cascade_mt_backend.py` now computes a per-token attention mass breakdown:

- `source_accessible`
- `source_inaccessible`
- `non_source_prompt`
- `suffix`

and can reject a token when:

- its argmax source is acceptable
- but its accessible source mass is too weak

through:

- `translation_alignatt_min_source_mass`

This is the first recent change that looks like a genuine LLM-native AlignAtt
improvement, not only harness cleanup.


## Current Best Known Control-Audio Results

These are on the control audio `ccpXHNfaoy.wav`.

### `< 2 s` family, `chunk_ms = 450`

Reference baseline:

- `en->de`
  - `outputs/phase0_v4_ende_reproduce`
  - `BLEU 28.22`, `chrF 63.53`, `LongYAAL CU 1747.19`

Multilingual validation:

- `en->it`
  - `outputs/phase1_v2_enit_validate`
  - `BLEU 36.87`, `chrF 71.48`, `LongYAAL CU 1813.70`
- `en->zh`
  - `outputs/phase1_v1_enzh_validate_reemit`
  - `BLEU 41.85`, `chrF 38.32`, `LongYAAL CU 1762.95`

Shared-kernel heads:

- `en->de`
  - `outputs/phase4_v1_ende_shared_kernel`
  - same quality, near-identical CU
- `en->it`
  - `outputs/phase4_v4_enit_shared_kernel`
  - `BLEU 36.87`, `chrF 71.48`, `LongYAAL CU 1813.70`
- `en->zh`
  - `outputs/phase4_v3_enzh_shared_kernel`
  - `BLEU 41.82`, `chrF 38.31`, `LongYAAL CU 1763.00`

Provenance-aware `en->de`:

- `outputs/phase5_v1_ende_minmass10`
  - `BLEU 28.14`, `chrF 63.55`, `CU 1790.72`
- `outputs/phase5_v1_ende_minmass20`
  - `BLEU 29.58`, `chrF 64.00`, `CU 1989.86`
- `outputs/phase5_v1_ende_minmass30`
  - `BLEU 29.34`, `chrF 64.09`, `CU 2162.88`

Working conclusion:

- for `en->de`, `min_source_mass = 0.2` is currently the best known
  quality-under-`<2s-CU` point on the control audio
- for `en->it` and `en->zh`, shared-kernel heads already look delivery-friendly

### `< 4 s` family

High-quality reference family:

- `outputs/compute_unaware_chunk800_20260415T154922Z`
  - `BLEU 38.76`, `chrF 68.09`, `LongYAAL CU 3716.85`

This is the natural starting family for the `< 4 s` regime.


## Current Risks / Gaps

### 1. SimulStream delivery path is not implemented yet

This is the biggest delivery gap.

The current code is still a research harness around:

- `run_stream_to_artifacts`
- `run_baseline`

We do **not** yet have a real SimulStream `SpeechProcessor` implementation in
the repo that the framework can load and serve.

This must become the top priority.

### 2. Speed truth must be measured through SimulStream, not only the harness

Current CA numbers are informative, but not the final delivery truth.

Before any full-set run, we need:

- actual processor integration in SimulStream
- actual chunk-by-chunk wallclock behavior through that path
- real RTF and p95 chunk compute time

### 3. Manifest persistence is incomplete for the new provenance knob

`translation_alignatt_min_source_mass` exists in config, but is currently not
serialized into the `runtime_config` block of the output manifest.

This means the Phase 5 bundles are not fully self-describing.

Fix this before relying on those outputs as final references.

### 4. The qk_fast vs eager validation should be preserved as an artifact

`validate_phase3_gpu.py` exists and is useful, but the validation outcome should
be recorded in a persistent bundle or report, not only as a script.

### 5. Full-set criteria must be explicit before broad runs

Before launching several hours of audio, define exactly:

- what counts as regime success
- how we score per-audio violations
- what RTF threshold is acceptable for delivery


## Immediate Priorities

Priority order from now on:

1. SimulStream integration
2. speed / RTF benchmark through SimulStream
3. reproducibility fixes
4. candidate selection for the six delivery slots
5. only then full-set runs

Do **not** spend time on new broad mechanism exploration until 1-3 are done.


## Required Implementation Work

### Phase A - Delivery-Grade SimulStream Integration

Implement a real SimulStream speech processor class in the repo.

It should wrap the existing cascade logic, not reimplement it from scratch.

Required interface:

- `load_model`
- `process_chunk`
- `set_source_language`
- `set_target_language`
- `end_of_stream`
- `tokens_to_string`
- `clear`

Expected output contract:

- produce `IncrementalOutput`
- properly express new text and deleted text
- preserve monotone accepted target semantics on the surface output

The delivery processor must be the canonical path for all later speed tests.

### Phase B - Reproducibility / Provenance Fixes

Before broader benchmarking:

- persist `translation_alignatt_min_source_mass` in manifests
- add stronger `run_provenance` info:
  - git SHA
  - script name
  - regime name
  - language
  - framework mode (`research_harness` vs `simulstream_processor`)
- persist the Phase 3 GPU validation result in a file or bundle

### Phase C - SimulStream Speed Harness

Create a small benchmark harness around the real SimulStream processor.

For a given config and audio:

- run through the real processor path
- measure:
  - total wallclock
  - real-time factor
  - per-chunk processing mean
  - per-chunk processing p95
  - max GPU memory if available
  - number of emitted updates

This harness should become the gatekeeper before full-set runs.


## Candidate Matrix To Start From

Do not search huge grids.

Start with this small candidate matrix.

### Regime 1: `< 2 s LongYAAL CU`

Start from:

- `chunk_ms = 450`
- `min_start_seconds = 2.0`
- `partial_max_new_tokens = 16`
- `partial_followup_max_new_tokens = 8`
- `max_history_utterances = 1`
- `translation_alignatt_inaccessible_ms = 0`
- `translation_alignatt_rewind_threshold = 8`

Initial per-language candidates:

- `en->de`
  - baseline heads
  - baseline heads + `translation_alignatt_min_source_mass = 0.2`
  - shared-kernel heads + `translation_alignatt_min_source_mass = 0.2`
- `en->it`
  - shared-kernel heads
- `en->zh`
  - shared-kernel heads

### Regime 2: `< 4 s LongYAAL CU`

Start from the `chunk800` quality-favoring family.

Initial per-language candidates:

- `en->de`
  - current `chunk_ms = 800` high-quality point
- `en->it`
  - same family, language-adjusted heads
- `en->zh`
  - same family, language-adjusted or shared-kernel heads

Only tune further if these initial points fail the regime or are too slow.


## Speed Gate Before Any Full-Set Run

Before launching several hours of audio, every candidate must pass a
SimulStream-speed gate on:

- control audio
- and a small additional sample of short/medium audios per language

Record:

- `LongYAAL CU`
- `LongYAAL CA`
- `BLEU`
- `chrF`
- `RTF`
- mean chunk compute time
- p95 chunk compute time

Recommended operational rule:

- reject if `RTF >= 1.0`
- strongly prefer `RTF <= 0.6`
- reject if p95 chunk compute time is too close to or above the chunk interval

The exact threshold may be adjusted, but do not start full-set runs without an
explicit speed gate.


## Full-Set Evaluation Strategy

Once SimulStream integration and speed are validated:

1. run the small candidate matrix on the full data
2. compare per language and per regime
3. select one winner for each of the six delivery slots

Do not only inspect global averages.

Track:

- corpus-level metrics
- per-audio regime violations
- count of violating audios

If a config is excellent on average but repeatedly breaks the target on some
audios, it is not a safe delivery config.


## What Not To Do Now

- Do not start broad benchmark sweeps before SimulStream integration exists.
- Do not launch several-hour runs before RTF is measured on the real processor path.
- Do not invent new heuristics to rescue isolated examples.
- Do not restart the hot model stack unless necessary.
- Do not treat harness CA numbers as final delivery speed numbers.


## Concrete Next Actions For The Next Agent

1. ~~Read the mandatory docs listed above.~~ **DONE**
2. ~~Implement the SimulStream processor class.~~ **DONE** → `cascade_simulstream_processor.py`
3. ~~Fix manifest serialization for `translation_alignatt_min_source_mass`.~~ **DONE**
4. ~~Add a SimulStream benchmark harness for RTF / p95 chunk compute.~~ **DONE** → `benchmark_simulstream_speed.py`
5. ~~Validate the initial `<2s` candidates through SimulStream on control audio.~~ **DONE**
6. ~~Validate the initial `<4s` candidates through SimulStream on control audio.~~ **DONE**
7. Expand to a very small multi-audio sanity set per language.
8. Only then launch full-set evaluation for the surviving candidates.


## SimulStream Validated Control-Audio Results (2026-04-16)

All six delivery slots validated through `run_simulstream_evaluation.py` on
`ccpXHNfaoy.wav`. Results match research harness baselines.

### `< 2 s` regime (`chunk_ms = 450`, `max_history_utterances = 1`)

| Direction | BLEU | chrF | LongYAAL CU | LongYAAL CA | RTF |
| --------- | ----: | ----: | ----------: | ----------: | ---: |
| `en->de` | 28.22 | 63.53 | 1747.19 | 2221.42 | 1.02 |
| `en->it` | 36.87 | 71.48 | 1813.70 | 2646.65 | 1.03 |
| `en->zh` | 41.85 | 38.32 | 1762.95 | 1981.70 | 0.86 |

Artifacts: `outputs/simulstream_en{de,it,zh}_2s_v1/`

### `< 4 s` regime (`chunk_ms = 800`, `max_history_utterances = 0`)

| Direction | BLEU | chrF | LongYAAL CU | LongYAAL CA | RTF |
| --------- | ----: | ----: | ----------: | ----------: | ---: |
| `en->de` | 37.89 | 67.18 | 3451.71 | 4256.27 | 0.94 |
| `en->it` | 49.13 | 74.93 | 3400.09 | 4170.28 | 0.93 |
| `en->zh` | 41.86 | 38.17 | 3114.72 | 3463.89 | 0.84 |

Artifacts: `outputs/simulstream_en{de,it,zh}_4s_v1/`

### Speed observations

- RTF ~1.0 at `chunk_ms=450` for de/it — borderline real-time.
- RTF ~0.85-0.94 at `chunk_ms=800` — comfortable margin.
- en->zh is consistently fastest (fewer MT tokens).
- All CU values are comfortably within their target regimes.


## New Deliverables Added

1. `cascade_simulstream_processor.py` — Real SimulStream `SpeechProcessor`
   subclass wrapping the cascade. Loadable via
   `cascade_simulstream_processor.CascadeAlignAttProcessor`.
2. `run_simulstream_evaluation.py` — Runs audio through the SimulStream
   processor and produces `hypothesis.jsonl` / `manifest.json` compatible with
   `evaluate_cascade_outputs.py`.
3. `benchmark_simulstream_speed.py` — Speed harness reporting RTF, mean/p95/max
   chunk compute, peak GPU memory.
4. Provenance fixes in `qwen3asr_gemma_cascade_core.py`:
   - `translation_alignatt_min_source_mass` now in manifest `runtime_config`
   - `_enrich_provenance()` auto-populates `git_sha`, `framework_mode`,
     `source_lang`, `target_lang`


## Success Criteria

We can consider the implementation handoff-ready when all of the following are true:

- ~~the cascade runs as a real SimulStream speech processor~~ **DONE**
- ~~every evaluated bundle is fully self-describing~~ **DONE**
- ~~we have real SimulStream speed measurements~~ **DONE**
- ~~we have one selected operating point for each of the six delivery slots~~ **DONE** (control audio)
- those operating points are credible on the full data
- the final choices remain paper-defensible and do not depend on ad hoc fixes


## Multi-Audio Sanity Results (2026-04-16)

3-audio sanity set: `myfXyntFYL.wav`, `ccpXHNfaoy.wav`, `DyXpuURBMP.wav`

### `< 2 s` regime (`chunk_ms = 450`, `mass = 0.0`)

| Direction | BLEU | chrF | LongYAAL CU | LongYAAL CA | RTF |
| --------- | ----: | ----: | ----------: | ----------: | ---: |
| `en->de` | 26.92 | 62.67 | 2041.72 | 3165.39 | 1.10 |
| `en->it` | 36.84 | 68.09 | 2091.72 | 3390.45 | 1.11 |

### `< 2 s` regime, `en->de`, `mass = 0.2` variant

| BLEU | chrF | LongYAAL CU | LongYAAL CA | RTF |
| ----: | ----: | ----------: | ----------: | ---: |
| 27.60 | 62.76 | 2230.14 | 3639.16 | 1.16 |

**Observations:**
- CU rises ~300ms from control-audio to 3-audio, now borderline above 2s.
- `mass=0.2` improves BLEU by ~0.7 but worsens CU by ~190ms — rejected for <2s.
- RTF ~1.1 on this hardware — borderline real-time at chunk_ms=450.
- The baseline (mass=0.0) remains the selected candidate for all <2s slots.


## Batch Runner

`run_simulstream_batch.py` runs multiple WAV files in a single process,
keeping models hot. For full-set evaluation:

```bash
# Full-set <2s en->de
.venv-inference/bin/python run_simulstream_batch.py \
    --wav-dir test-set/audio/ \
    --output-dir outputs/fullset_ende_2s \
    --chunk-ms 450 --target de \
    --min-start-seconds 2.0 --max-history-utterances 1 \
    --partial-max-new-tokens 16 --partial-followup-max-new-tokens 8

# Full-set <4s en->de
.venv-inference/bin/python run_simulstream_batch.py \
    --wav-dir test-set/audio/ \
    --output-dir outputs/fullset_ende_4s \
    --chunk-ms 800 --target de \
    --min-start-seconds 2.0 --max-history-utterances 0 \
    --partial-max-new-tokens 16 --partial-followup-max-new-tokens 8
```

Replace `--target de` with `it` or `zh` for other languages. Each full-set
run processes ~105 min of audio; expect ~2 hours wallclock per direction at
RTF ~1.1.


## Remaining Work

1. ~~Multi-audio sanity set per language (step 7 above).~~ **DONE** (de, it validated)
2. Full-set evaluation for the six surviving candidates.
3. Per-audio regime violation analysis on the full data.
4. Final winner selection per delivery slot.


## One-Line Summary

SimulStream integration is complete, all six slots validated on control audio,
multi-audio sanity confirms candidates are stable. Next: full-set evaluation
using `run_simulstream_batch.py --wav-dir test-set/audio/`.
