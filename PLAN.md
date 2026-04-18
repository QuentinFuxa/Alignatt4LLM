# Current Plan

Date: 2026-04-18

This file is the active overnight runbook. Historical broad planning belongs in
`docs/archive/`; session-level decisions still belong in `DECISIONS.md`.

## Current validated point

Runtime surface:

- ASR: `qwen_forced`
- MT: `gemma_vllm_alignatt`
- Public output: append-only exact string surface
- Current low-latency working point: `chunk_ms=750`,
  `translation_alignatt_border_margin=1`

Known full-dev-set results at this point:

- `en->de` in `outputs/iwslt26_devset_chunk750_borderp1_ende`
  - BLEU `27.3549`
  - chrF `61.4566`
  - COMET `0.8669`
  - LongYAAL (CU) `1707.1609 ms`
  - Empty predictions `0 / 919`
- `en->it` in `outputs/iwslt26_devset_chunk750_borderp1_enit`
  - BLEU `38.3665`
  - chrF `66.8151`
  - COMET `0.7875`
  - LongYAAL (CU) `1675.4422 ms`
  - Empty predictions `0 / 919`

`en->zh` is the missing direction for this exact low-latency point.

## Global priorities

1. Finish the current low-latency matrix by running and scoring `en->zh`.
2. Find the best **LONG regime** configuration (`LongYAAL CU < 4 s`) with a
   clean, paper-defensible tuning procedure.
3. Freeze the chosen configs into the submission surface and validate the
   Docker image / log format expected by IWSLT.
4. Return to the `< 2 s` regime and move quality upward while staying safely
   under the latency budget.

## Operating rules

- Use `.venv-inference` for inference and `.venv-evaluation` for OmniSTEval.
- Keep runs sequential on one GPU. Do not run two full cascades in parallel.
- Prefer GPU scoring for COMET, but do not score concurrently with inference.
- Treat full-dev-set runs as expensive. Use one real audio first, then two, and
  only then promote to full dev set.
- Use the IWSLT-compatible env for all official-like runs:
  - `VLLM_USE_DEEP_GEMM=0`
  - `VLLM_MOE_USE_DEEP_GEMM=0`

## Phase 0 — Complete the current low-latency matrix

Goal: obtain the missing `en->zh` dev-set result for the already validated
`750 ms / border_margin=1` point.

Steps:

1. Run full dev set:

```bash
VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
.venv-inference/bin/python run_simulstream_batch.py \
  --input-dir dev-set/audio \
  --output-dir outputs/iwslt26_devset_chunk750_borderp1_enzh \
  --chunk-ms 750 \
  --source en \
  --target zh \
  --translation-alignatt-border-margin 1
```

2. Score on GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
.venv-evaluation/bin/python evaluate_cascade_outputs.py \
  --output-dir outputs/iwslt26_devset_chunk750_borderp1_enzh
```

Acceptance criteria:

- `Empty Predictions = 0`
- no obvious emission collapse on the long talks
- LongYAAL (CU) remains safely below `2 s`

## Phase 1 — Determine the best LONG-regime config (`< 4 s`)

### Objective

Maximize quality while staying in the long regime with safety margin on real
talks. Do not jump straight to full-dev sweeps.

### Pilot procedure

Use these two real talks, in this order:

1. `dev-set/audio/OiqEWDVtWk.wav`
   - historically exposed fragile frontier / append-only behavior
2. `dev-set/audio/ccpXHNfaoy.wav`
   - stable, long, and already well-characterized

Use `en->de` as the pilot direction first. Once a clear winner exists on
`en->de`, confirm it on `en->it` and `en->zh`.

### Knobs to tune

Tune in this order, and keep the search narrow:

1. `chunk_ms` — primary long-regime knob
2. `translation_alignatt_inaccessible_ms` — secondary, time-domain source
   border margin; easiest to defend in a paper
3. `translation_alignatt_border_margin` — only if needed after the first two
   axes; keep the sweep very small

Do **not** start with `translation_alignatt_min_source_mass` or
`translation_alignatt_argmax_mass_threshold`. Those are useful research knobs,
but they are higher risk and should be touched only if the simpler surfaces
fail to separate candidates.

### Recommended search

Stage A: coarse chunk sweep on one talk, keep AlignAtt otherwise fixed

- `chunk_ms in {900, 1100, 1300, 1500}`
- `translation_alignatt_border_margin=1`
- `translation_alignatt_inaccessible_ms=0`

If everything is still comfortably below `3.2 s` on `OiqEWDVtWk.wav`, allow one
extra point at `1700 ms`. If any point becomes clearly sparse or crosses `4 s`,
drop larger chunks immediately.

Stage B: refine with a time-domain source-border buffer

- around the best `chunk_ms`, test
  `translation_alignatt_inaccessible_ms in {0, 150, 300}`

Only if needed, Stage C: tiny token-domain correction

- test `translation_alignatt_border_margin in {0, 1}`

### Promotion rule

For each candidate, require:

- `Empty Predictions = 0`
- dense emission throughout the talk
- no late freeze on `OiqEWDVtWk.wav`
- LongYAAL (CU) comfortably under `4 s`

Promote at most **two** candidates to the next stage:

1. best two on `OiqEWDVtWk.wav`
2. rerun those two on `ccpXHNfaoy.wav`
3. run full dev set on `en->de` only
4. freeze one winner
5. run that winner on `en->it` and `en->zh`

### What “best” means

Ranking priority:

1. `COMET`
2. `chrF`
3. `BLEU`
4. lowest latency, only among quality-tied candidates

Interpret latency as a constraint, not the main objective, for this phase.
Being at `3.3-3.8 s` is acceptable if quality gains are real and the run stays
clean.

## Phase 2 — Freeze the submission surface and build Docker

Goal: turn the chosen configs into the actual IWSLT-facing submission path.

Steps:

1. Encode the final presets in `cascade_submission.py`.
   - update the existing frozen presets if the choice is final
   - otherwise add clearly named temporary presets first, validate them, then
     rename only once they are frozen
2. Update `submission/README.md` and the top-level `README.md` so the documented
   preset names and chunk sizes match reality.
3. Build the container:

```bash
docker build -t cascade-simul-iwslt26 .
```

4. Dry-run the container on a tiny wavlist following `submission/README.md`:

```bash
docker run --gpus all --rm \
  -e PRESET=<chosen_preset> \
  -e SRC_LANG=English \
  -e TGT_LANG=German \
  -e SRC_LANG_CODE=en \
  -e TGT_LANG_CODE=de \
  -v /host/wavs:/io/wavs:ro \
  -v /host/out:/io/out \
  cascade-simul-iwslt26 \
  /io/wavs/wavlist.txt /io/out/metrics.jsonl
```

5. Verify:
   - the container starts without manual flags
   - `metrics.jsonl` is written
   - output format matches the SimulStream/IWSLT expectation
   - the preset baked into Docker matches the intended regime

## Phase 3 — Return to the `< 2 s` regime and move upward

Goal: stay under `LongYAAL CU = 2 s` while pushing quality closer to the
latency ceiling.

Start from the validated point:

- `chunk_ms=750`
- `translation_alignatt_border_margin=1`
- `translation_alignatt_inaccessible_ms=0`

Recommended search:

1. Tune `chunk_ms` first:
   - `{775, 800, 825}`
   - if still clearly below `2 s`, allow `850`
2. Keep `border_margin=1` fixed initially
3. Only if needed, test `translation_alignatt_inaccessible_ms in {0, 100}`

Procedure:

1. pilot on `OiqEWDVtWk.wav`
2. confirm on `ccpXHNfaoy.wav`
3. full dev set on `en->de`
4. then `en->it` and `en->zh`

Success criterion:

- stay below `2 s` on full-dev `LongYAAL (CU)`
- `Empty Predictions = 0`
- quality improves over the current `750 / +1` baseline

## Short checklist for the next agent

- Finish `en->zh` at the current low-latency point first.
- Do the LONG-regime search on one real talk, then two, then full dev.
- Tune `chunk_ms` first; keep the rest narrow and principled.
- Freeze the chosen config into `cascade_submission.py`.
- Build and dry-run Docker before touching more tuning.
- Only then come back to the `< 2 s` regime and try to spend the remaining
  latency budget for extra quality.
