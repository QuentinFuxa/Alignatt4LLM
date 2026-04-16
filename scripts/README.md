# `scripts/`

Dated research scripts preserved for reference. **Not maintained** — they may or may not run against the current `cascade_runtime.py` surface.

The canonical, maintained entry points are at the repo root:

- `run_simulstream_batch.py` — canonical streaming evaluation runner
- `run_simulstream_compare.py` — A/B two alignment backends
- `run_alignment_single_audio.py` — ASR-side diagnostic harness
- `run_mt_backend_parity.py` — MT backend parity harness
- `evaluate_cascade_outputs.py` — OmniSTEval + XCOMET-XL evaluation

## How to run one of these historical scripts

They import from the repo root (`cascade_runtime`, `cascade_mt_backend`, `qwen3asr_gemma_cascade_core`, etc.). Run them with the repo root on `PYTHONPATH`:

```bash
cd /home/cascade_simultaneous
PYTHONPATH=. .venv-inference/bin/python scripts/<name>.py [args]
```

Without `PYTHONPATH=.`, the imports will fail because Python only puts the script's own directory on `sys.path` by default.

## Inventory (as of 2026-04-16)

| Script | Historical purpose |
|---|---|
| `analyze_cascade_timings.py` | Aggregate timing breakdown from `stream_updates.jsonl`. Pre-`full_vllm` era. |
| `benchmark_simulstream_speed.py` | Per-chunk latency harness; reports RTF, mean/p95/max chunk ms, peak GPU. |
| `build_alignatt_head_set.py` | Materialise a head-set regime (shared kernel, multilingual union) to disk. |
| `qwen3asr_gemma_cascade_notebook.py` | Jupyter-oriented cascade driver that wraps `qwen3asr_gemma_cascade_core`. |
| `reemit_cascade_outputs.py` | Re-emit artifacts from an existing run with different emission params. |
| `restart_jupyter_kernel.py` | List/control/start/warm notebook kernels when running in Jupyter. |
| `run_cascade_baseline.py` | One-shot baseline run through `qwen3asr_gemma_cascade_core.run_baseline`. |
| `run_gemma_asr_fairness.py` | Controlled ablation of a Gemma-ASR WER discrepancy. |
| `run_latency_experiment.py` | Warm-kernel single-audio latency runs with module hot-reload. |
| `run_phase0_reproduction.py` | Reproduce PLAN operating point on the control audio (was `scripts_run_phase0_reproduction.py`). |
| `run_phase2_caps.py` | Probe `partial_max_new_tokens` lever at fixed operating point (was `scripts_run_phase2_caps.py`). |
| `run_stage1_start_gate_sweep.py` | `min_start_seconds` sweep. |
| `run_streaming_stability.py` | Per-chunk alignment stability diagnostic. |
| `standalone_gemma_asr_test.py` | Gemma ASR smoke test using `AutoModelForMultimodalLM`. |
| `validate_phase3_gpu.py` | Validate qk_fast vs eager agreement and prefix-online invariants on real GPU. |
| `validate_qk_fast_audio.py` | Validate qk_fast vs eager for Gemma audio forced alignment. |

## What depends on `qwen3asr_gemma_cascade_core.py`

That file is a **compatibility shim** over `cascade_runtime.py` (see `AGENTS.md`). It is kept at the repo root for these scripts to import; no active code imports it. If you remove it, the following scripts stop working:

- `run_cascade_baseline.py`
- `run_latency_experiment.py`
- `run_stage1_start_gate_sweep.py`
- `run_phase0_reproduction.py`
- `run_phase2_caps.py`
- `qwen3asr_gemma_cascade_notebook.py`
