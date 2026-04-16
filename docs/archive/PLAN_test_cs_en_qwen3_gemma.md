# Plan: Test The Qwen3-ASR + Gemma Cascade On 3 English->Czech Audios

## Goal

Run a focused `en->cs` smoke evaluation of the current Qwen3-ASR + Gemma cascade on exactly `3` audios, keeping models hot and producing artifacts another agent can inspect.

This is a test plan, not a promise that the current repo is already fully wired for `en->cs`.

## Why This Needs A Plan

Two repo-state blockers exist right now:

1. The runtime auto-resolves text AlignAtt heads via:

   - `assets/attention_heads/translation_heads_google_gemma-4-E4B-it_<source>-<target>.json`

   For `en->cs`, that means it expects:

   - `assets/attention_heads/translation_heads_google_gemma-4-E4B-it_en-cs.json`

   That file does **not** exist yet.

2. Local OmniSTEval-style evaluation is not ready for Czech because the repo does **not** contain:

   - `test-set/ref/cs.txt`

   So the other agent can run inference on 3 audios immediately, but full local Czech-reference evaluation is blocked until a proper Czech reference file aligned with `test-set/audio-segments.yaml` exists.

## Recommended Test Set

Use the same 3-audio sanity set already suggested by `run_simulstream_batch.py`:

- `test-set/audio/myfXyntFYL.wav`
- `test-set/audio/DyXpuURBMP.wav`
- `test-set/audio/ccpXHNfaoy.wav`

This keeps the test grounded in an existing repo convention rather than inventing a new trio.

## Backend Choice

For the `qwen3_gemma` test, use:

- ASR / alignment backend: `qwen_forced`
- MT backend: Gemma through the normal cascade runtime

This is the stable path in the current repo. Do **not** start with `gemma_onepass_qk_fast` for this first `en->cs` 3-audio test.

## Preconditions

### 1. Use The Inference Environment

Run from the shared inference env:

```bash
../iwslt-2026-baselines/.venv-inference/bin/python --version
```

### 2. Ensure GPU Is Clean

```bash
nvidia-smi
pkill -f 'run_simulstream_batch.py|run_simulstream_compare.py|VLLM::EngineCore' || true
```

Only continue if the GPU is effectively free.

### 3. Prepare An `en->cs` Heads File

Preferred option:

- use a real `en->cs` heads file if one has already been calibrated.

Expected path:

- `assets/attention_heads/translation_heads_google_gemma-4-E4B-it_en-cs.json`

If that file does not exist yet, use a clearly marked **provisional multilingual prior** built from the existing shared kernel.

Suggested temporary materialization:

```bash
python3 - <<'PY'
import json
from pathlib import Path

src = Path("assets/attention_heads/translation_heads_shared_kernel_top8.json")
dst = Path("assets/attention_heads/translation_heads_google_gemma-4-E4B-it_en-cs.json")

payload = json.loads(src.read_text(encoding="utf-8"))
payload["direction"] = "en-cs"
payload["provenance_note"] = (
    "Provisional en-cs test-time head bundle derived from the multilingual "
    "shared kernel. Not a true en-cs calibrated artifact."
)

dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(dst)
PY
```

Important:

- This fallback is acceptable for a smoke test.
- It must be reported explicitly as provisional.
- It must **not** be misrepresented as a calibrated `en->cs` head set.

## Phase 1: Single-Audio Sanity Check

Do **not** jump directly to 3 audios before confirming that `en->cs` actually runs.

Run:

```bash
../iwslt-2026-baselines/.venv-inference/bin/python run_simulstream_batch.py \
  --wavs test-set/audio/ccpXHNfaoy.wav \
  --output-dir outputs/simulstream_en_cs_smoke1 \
  --chunk-ms 450 \
  --source en \
  --target cs \
  --alignment-backend-name qwen_forced \
  --min-start-seconds 2.0 \
  --max-history-utterances 1 \
  --partial-max-new-tokens 16 \
  --partial-followup-max-new-tokens 8 \
  --translation-alignatt-inaccessible-ms 0.0 \
  --translation-alignatt-rewind-threshold 8 \
  --translation-alignatt-min-source-mass 0.0
```

### Phase-1 Pass Criteria

Only continue to the 3-audio batch if all of the following are true:

- the run completes without crashing
- `outputs/simulstream_en_cs_smoke1/manifest.json` exists
- `outputs/simulstream_en_cs_smoke1/hypothesis.jsonl` exists
- the manifest says:
  - `source_language_code = "en"`
  - `target_language_code = "cs"`
- `runtime_config.translation_alignatt_heads_path` in the manifest points to the intended `en-cs` file
- the prediction is actually Czech-like, not English copy or German/Italian leakage

### Phase-1 Stop Conditions

Stop immediately and do **not** run the 3-audio batch if any of these happen:

- missing `en-cs` heads path
- runtime silently loads the wrong heads file
- output remains mostly English copying
- the model crashes or OOMs

## Phase 2: 3-Audio Batch Test

If the single-audio sanity check passes, run the full 3-audio batch:

```bash
../iwslt-2026-baselines/.venv-inference/bin/python run_simulstream_batch.py \
  --wavs \
    test-set/audio/myfXyntFYL.wav \
    test-set/audio/DyXpuURBMP.wav \
    test-set/audio/ccpXHNfaoy.wav \
  --output-dir outputs/simulstream_en_cs_qwen3_gemma_3audio \
  --chunk-ms 450 \
  --source en \
  --target cs \
  --alignment-backend-name qwen_forced \
  --min-start-seconds 2.0 \
  --max-history-utterances 1 \
  --partial-max-new-tokens 16 \
  --partial-followup-max-new-tokens 8 \
  --translation-alignatt-inaccessible-ms 0.0 \
  --translation-alignatt-rewind-threshold 8 \
  --translation-alignatt-min-source-mass 0.0
```

## Artifacts To Collect

After the batch run, save and report:

- `outputs/simulstream_en_cs_qwen3_gemma_3audio/manifest.json`
- `outputs/simulstream_en_cs_qwen3_gemma_3audio/hypothesis.jsonl`
- `outputs/simulstream_en_cs_qwen3_gemma_3audio/stream_updates.jsonl`

The batch runner does not emit per-audio `translation.cs.txt` files, so the main review surface is `hypothesis.jsonl`.

## Manual Review Checklist

The other agent should report all of the following:

- which heads file was used:
  - real calibrated `en-cs`
  - or provisional multilingual shared-kernel fallback
- whether the run completed on all 3 audios
- `batch_rtf`
- per-audio `rtf`
- per-audio update count
- whether outputs look Czech and fluent
- whether outputs appear too literal, too delayed, or prone to copy-through
- whether any audio produced pathological rewrites or instability

## Local Evaluation Status

### What Is Possible Immediately

Immediate local testing can cover:

- inference success / failure
- speed
- update behavior
- qualitative Czech output inspection

### What Is Blocked Locally

Do **not** try to run `evaluate_cascade_outputs.py` for `en->cs` until a Czech reference file exists and is aligned with the repo segmentation.

Current blocker:

- `test-set/ref/cs.txt` is missing

To unlock OmniSTEval later, another agent would need a proper Czech reference file compatible with:

- `test-set/audio-segments.yaml`

Then the evaluation command would be:

```bash
python evaluate_cascade_outputs.py \
  --output-dir outputs/simulstream_en_cs_qwen3_gemma_3audio \
  --target-lang-code cs \
  --target-reference /path/to/full/test-set/ref/cs.txt \
  --skip-comet
```

## Expected Deliverable From The Testing Agent

The other agent should return:

1. whether `en->cs` ran successfully on 1 audio
2. whether the 3-audio batch ran successfully
3. which heads bundle was used
4. the manifest snippet showing the effective `translation_alignatt_heads_path`
5. the 3 final Czech predictions from `hypothesis.jsonl`
6. a short verdict:
   - runnable but provisional
   - runnable and promising
   - blocked

## Decision Rule

Interpret the outcome as follows:

- If the smoke run fails before producing Czech output, the blocker is structural and must be fixed before any broader test.
- If the smoke run works but requires the provisional shared-kernel fallback, report the result as exploratory rather than calibrated.
- If the 3-audio batch works and the Czech outputs are qualitatively sane, the next step is to calibrate or import a real `en->cs` heads bundle and then rerun the same 3-audio protocol.
