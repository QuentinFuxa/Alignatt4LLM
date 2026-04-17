# cascade_simultaneous

Research repo for **simultaneous speech translation** via an ASR → MT cascade with **AlignAtt**-based emission control on both sides.

## Current recommended combination

```
alignment_backend_name = "qwen_forced"          # Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B
mt_backend_name        = "gemma_vllm_alignatt"  # Gemma-4-E4B MT through vLLM + engine-native AlignAtt observer
```

Full runtime matrix and rationale: [`docs/RUNTIME_ARCHITECTURE.md`](docs/RUNTIME_ARCHITECTURE.md).

## Canonical validation loop

```bash
# inference
.venv-inference/bin/python run_simulstream_batch.py \
    --alignment-backend-name qwen_forced \
    --mt-backend-name gemma_vllm_alignatt \
    --inputs test-set/audio/ccpXHNfaoy.wav \
    --output-dir outputs/my_run

# evaluation (BLEU / chrF / XCOMET-XL / LongYAAL CU / LongYAAL CA)
.venv-evaluation/bin/python evaluate_cascade_outputs.py \
    --output-dir outputs/my_run
```

Reference numbers (`test-set/audio/ccpXHNfaoy.wav`, 360 s, en→de, chunk_ms=450): **BLEU 27.5, chrF 63.5, COMET 0.861, LongYAAL CU 1766 ms, LongYAAL CA 1473 ms, RTF 0.401.** Full calibration curve and multi-clip numbers in [`docs/RESULTS.md`](docs/RESULTS.md).

## Docs map

- [`PLAN.md`](PLAN.md) — current plan and next steps
- [`DECISIONS.md`](DECISIONS.md) — append-only log of session-level decisions and what changed why
- [`docs/RUNTIME_ARCHITECTURE.md`](docs/RUNTIME_ARCHITECTURE.md) — ASR/MT axes, module map, session lifecycle
- [`docs/MT_VLLM_BACKEND.md`](docs/MT_VLLM_BACKEND.md) — design of the experimental `gemma_vllm_alignatt` MT backend (Phases 0–5)
- [`docs/RESULTS.md`](docs/RESULTS.md) — consolidated quality/latency numbers + calibration curve
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — GPU / vLLM / allocator gotchas
- [`docs/CONTEXT_INJECTION.md`](docs/CONTEXT_INJECTION.md) — ACL-paper extra-context mechanism (IWSLT 2026 sub-track)
- [`submission/README.md`](submission/README.md) — Docker/log submission entry points and frozen presets
- [`docs/archive/`](docs/archive/) — historical design docs preserved for context
- [`docs/reference/`](docs/reference/) — upstream model cards and reference papers/implementations
- [`AGENTS.md`](AGENTS.md), [`CLAUDE.md`](CLAUDE.md) — operational guidance for agents

## Entry points at the repo root

| Script | Purpose |
|---|---|
| `run_simulstream_batch.py` | **Canonical runner.** Streaming evaluation on one or many media files (`.wav`, `.mp4`, ...). |
| `run_simulstream_compare.py` | A/B two alignment backends on one clip. |
| `run_iwslt_submission.py` | Frozen IWSLT submission presets for offline logs or websocket serving. |
| `run_alignment_single_audio.py` | ASR-side diagnostic harness. |
| `run_mt_backend_parity.py` | MT backend parity harness (Transformers vs vLLM, subprocess-isolated). |
| `run_context_ablation.py` | Three-condition MT paper-context ablation on one clip (hot bundle). |
| `evaluate_cascade_outputs.py` | BLEU / chrF / XCOMET-XL / LongYAAL over a run bundle. |

All other one-off research scripts live under [`scripts/`](scripts/).

## Extra-context (IWSLT 2026 sub-track)

ACL-paper PDFs can be preprocessed into compact `PaperArtifact` JSON and injected into the Gemma MT prompt as a `[Paper context]` block (retrieved top-k chunks via BM25 or static title+abstract). The runtime pairs with a **provenance guard** (`translation_alignatt_min_source_mass`) that vetoes any drafted token whose MT attention concentrates on the paper-context span, eliminating paper-content leakage observed with naive context injection.

Design, empirical ablations, and the submission-defensible setting live in [`docs/CONTEXT_INJECTION.md`](docs/CONTEXT_INJECTION.md). Default is `paper_context_mode=off` — extra-context injection is fully opt-in.

## Environments

- `.venv-inference` — all inference and streaming paths
- `.venv-evaluation` — OmniSTEval + XCOMET-XL (do not mix with the inference env)

## Tests

```bash
.venv-inference/bin/python -m pytest test_*.py -q
```

No test needs a model to be loaded; every GPU-shaped test uses synthetic payloads or config-only dispatch checks. Live GPU validation goes through the runners above with `--wavs` or `--wav-dir`.
