# AlignAtt4LLM

Research code for simultaneous speech translation with a streaming ASR-to-MT
cascade and decoder-only AlignAtt policies.

Paper: [AlignAtt4LLM: Fast AlignAtt for Decoder-Only LLMs at IWSLT 2026
Simultaneous Speech Translation Task](https://arxiv.org/abs/2606.03967).

This branch is the public research-code surface. The paper source and generated
PDF are not vendored here; the arXiv record is the canonical paper artifact.

## Active Focus

- Improve EN->ZH by using `milmmt_vllm_alignatt` instead of the Gemma MT route.
- Produce stronger, reproducible evidence that AlignAtt beats fixed
  `cut_last_target_units` policies under comparable latency.
- Keep the runtime clean enough that new results can become paper-grade figures
  and tables without another repo cleanup pass.

## Public Scope

- Included: active runtime code, presets, AlignAtt head payloads, references,
  evaluation/reporting utilities, and post-submission research notes.
- Not included: paper LaTeX/PDF sources, Docker submission packaging, local
  experiment outputs, model weights, and audio files whose redistribution is
  controlled by upstream datasets.
- Runtime target: a CUDA/A100-style inference machine with the vLLM stack used
  by this project. Lightweight inspection and policy tests can run without GPU
  inference.

## Runtime

- ASR default: `qwen_forced`
- MT baseline: `gemma_vllm_alignatt`
- MT improvement route: `milmmt_vllm_alignatt`
- Active presets: `gemma_low_latency`, `gemma_high_latency`
- Preset API: `cascade.presets.get_runtime_preset`
- Batch runner: `run_simulstream_batch.py`
- Single-audio A/B: `run_simulstream_compare.py`

## Setup

Inference environment:

```bash
./setup_inference_qwen_asr_vllm.sh
```

Evaluation environment:

```bash
uv venv .venv-evaluation --python 3.13
UV_PROJECT_ENVIRONMENT=.venv-evaluation uv sync --group evaluation
```

## Data

The repository tracks dev-set metadata/references and small text fixtures under
`data/`. Audio is expected to be provided locally by the user, typically under
`data/devset/audio/`, and is ignored by Git.

For the historical smoke path used in internal notes,
`data/smoke/alignatt_smoke18.wav` is a local 18-second clip derived from the
dev-set audio. It is not part of the public Git payload.

## Canonical Commands

Smoke the ASR frontend comparison on one local clip:

```bash
.venv-inference/bin/python run_simulstream_compare.py \
  --wav <path-to-local-wav>
```

Run the Gemma baseline on one dev clip:

```bash
.venv-inference/bin/python run_simulstream_batch.py \
  --inputs <path-to-local-wav> \
  --target de \
  --output-dir outputs/gemma_de_smoke
```

Probe the MiLMMT EN->ZH route:

```bash
.venv-inference/bin/python run_simulstream_batch.py \
  --inputs <path-to-local-wav> \
  --target zh \
  --mt-backend-name milmmt_vllm_alignatt \
  --translation-alignatt-top-k-heads 8 \
  --output-dir outputs/milmmt_zh_smoke
```

Compare AlignAtt against fixed target cutoffs on a controlled subset:

```bash
.venv-inference/bin/python scripts/run_mt_cutoff_policy_sweep.py \
  --inputs <path-to-local-wav> \
  --target de \
  --output-root outputs/mt_cutoff_smoke
```

Score an output directory:

```bash
.venv-evaluation/bin/python evaluate_cascade_outputs.py \
  --output-dir outputs/gemma_de_smoke
```

## Repo Layout

- `cascade/` — active runtime package
- `cascade/presets.py` — active runtime preset source of truth
- `data/devset/` — tracked development metadata and references
- `dev-set/` — compatibility alias to `data/devset/`
- `data/alignatt_heads/` — tracked AlignAtt head payloads
- `data/smoke/` — tiny text fixtures for local smoke runs
- `scripts/` — maintained experiment and reporting utilities
- `docs/` — current system, status, and result notes
- `docs/archive/` — historical cleanup and submission notes

## Docs

- [`docs/system.md`](docs/system.md) — runtime architecture and supported routes
- [`docs/results.md`](docs/results.md) — result anchors and calibration notes
- [`docs/status.md`](docs/status.md) — current cleanup state and decisions
- [`docs/archive/2026-05-submission.md`](docs/archive/2026-05-submission.md) —
  compact record of the submitted Docker/preset era
- [`docs/reference/README.md`](docs/reference/README.md) — retained upstream reading material

## License

Code in this repository is released under the MIT License. See [`LICENSE`](LICENSE).
