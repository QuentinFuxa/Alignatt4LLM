# cascade_simultaneous

Research repo for simultaneous speech translation with a streaming
ASR-to-MT cascade.

## Current runtime

- ASR default: `qwen_forced`
- MT default: `gemma_vllm_alignatt`
- Experimental MT route: `milmmt_vllm_alignatt` (opt-in only)
- Canonical runner: `run_simulstream_batch.py`
- Canonical single-audio A/B: `run_simulstream_compare.py`
- Submission presets: `main_low_latency` = `chunk_ms=850`, `main_high_latency` = `chunk_ms=1500`
- Docker submission directions: EN->DE, EN->IT, EN->ZH only
- Official organizer baseline: `https://github.com/owaski/iwslt-2026-baselines`
  with score extraction via `scripts/parse_official_baseline_outputs.py`
- MT AlignAtt no longer uses an anti-rewind threshold; legitimate EN->ZH reorderings
  make that heuristic a bad fit for streaming MT.

## Frozen MT AlignAtt policy

The maintained MT policy is a source-frontier monitor, not a lexical repair
rule and not a fixed target delay. For each partial Gemma draft, the runtime
replays compact attention diagnostics on retained MT heads and emits only the
append-only prefix whose attention looks supported by the currently accessible
source.

The frozen low-latency preset is:

- `chunk_ms=850`
- `translation_alignatt_top_k_heads=4`
- `translation_alignatt_border_margin=1`
- `translation_alignatt_min_source_mass=0.003`
- `translation_alignatt_frontier_min_inaccessible_mass=0.03`
- `translation_alignatt_max_inaccessible_source_mass=0.15`

What the policy monitors:

- The selected-head attention row for each candidate target token.
- The token's argmax source position, mapped back to source stability units.
- The current source frontier: which source units are accessible at this audio
  time.
- Total attention mass on inaccessible source positions (`future mass`).
- A tiny accessible-source mass floor, used only as a sanity check.

Why this shape: fixed `cut-last-x` policies delay every target draft by the
same amount, even when the next target token only depends on already heard
source. AlignAtt instead spends latency conditionally. It allows small frontier
crossings when future-source mass is negligible, but blocks a token when the
attention distribution carries meaningful mass beyond the source frontier. The
0.003 accessible-source floor prevents completely ungrounded emissions without
requiring every function word or morphology token to have strong source mass.

On the full 21-audio EN->DE dev set, this frozen point stays below the 2 s
CU-LongYAAL budget with solid quality. The three-clip diagnostic reported in
the paper uses the same policy family and shows why attention can beat fixed
`cut-last-x` rules under the same latency budget.

## Canonical commands

```bash
.venv-inference/bin/python run_simulstream_batch.py \
  --inputs data/devset/audio/ccpXHNfaoy.wav \
  --output-dir outputs/my_run

.venv-evaluation/bin/python evaluate_cascade_outputs.py \
  --output-dir outputs/my_run
```

## Docker submission

Build on an NVIDIA H100 host with a Hugging Face token available as a BuildKit
secret. The maintained helper builds, optionally validates on one WAV, and pushes
both a commit tag and `latest`:

```bash
export DOCKERHUB_REPO="dockerhub-user/cascade-simul-iwslt26"
export HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
submission/build_push_dockerhub_h100.sh
```

Set `PUSH=0` to stop after the local build.

Direct inference is the default container mode:

```bash
docker run --gpus all --rm --ipc=host \
  -e PRESET=main_low_latency \
  -e TGT_LANG_CODE=de \
  -v /host/wavs:/io/wavs:ro \
  -v /host/out:/io/out \
  "$DOCKERHUB_REPO:latest" \
  infer /io/wavs/wavlist.txt /io/out/metrics.jsonl
```

The same image can expose a SimulStream HTTP speech processor:

```bash
docker run --gpus all --rm --ipc=host -p 8080:8080 \
  -e PRESET=main_low_latency \
  -e TGT_LANG_CODE=de \
  "$DOCKERHUB_REPO:latest" serve
```

## Repo layout

- `cascade/` — active runtime package
- `data/devset/` — tracked development set and references
- `dev-set/` — compatibility alias to `data/devset/`
- `data/alignatt_heads/` — tracked AlignAtt head payloads used by runtime and paper tooling
- `data/smoke/` — tiny reproducible smoke fixtures
- `docs/` — current system, results, status, and submission docs
- `scripts/` — maintained utility scripts only
- `submission/` — Docker submission surface
- `paper/` — paper sources and retained generated TeX fragments

## Docs

- [`docs/system.md`](docs/system.md) — runtime architecture, supported backends, operational notes
- [`docs/results.md`](docs/results.md) — consolidated calibration and reference numbers
- [`docs/status.md`](docs/status.md) — current repo status and cleanup decisions
- [`docs/submission.md`](docs/submission.md) — submission workflow and bundle/export story
- [`submission/README.md`](submission/README.md) — concrete submission workspace usage
- [`docs/reference/README.md`](docs/reference/README.md) — retained upstream reading material
