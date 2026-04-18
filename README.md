# cascade_simultaneous

Research repo for **simultaneous speech translation** via an ASR → MT cascade with word-timed ASR and **MT-side AlignAtt** emission control.

## Current Runtime Surface

```
alignment_backend_name = "qwen_forced"          # Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B
mt_backend_name        = "gemma_vllm_alignatt"  # Gemma-4-E4B MT through vLLM + qk_fast AlignAtt observer
ASR commit path        = punctuation_lcp + EOS flush   # fixed in the current worktree
ASR -> MT conditioning = full live ASR tail, with unstable trailing sentence punctuation stripped
MT emission limit      = AlignAtt acceptance over that full MT-visible source prefix
run_simulstream_batch  = chunk_ms=800, max_history_utterances=0 by default
submission presets     = main_low_latency: chunk_ms=750 border_margin=1 (LOW regime), main_high_latency: chunk_ms=1100 border_margin=1 (HIGH regime)
```

Full runtime matrix and the exact shipped surfaces: [`docs/RUNTIME_ARCHITECTURE.md`](docs/RUNTIME_ARCHITECTURE.md).

## Canonical validation loop

```bash
# inference
.venv-inference/bin/python run_simulstream_batch.py \
    --inputs dev-set/audio/ccpXHNfaoy.wav \
    --output-dir outputs/my_run

# evaluation (BLEU / chrF / XCOMET-XL / LongYAAL CU / LongYAAL CA)
.venv-evaluation/bin/python evaluate_cascade_outputs.py \
    --output-dir outputs/my_run
```

Historical reference numbers (`dev-set/audio/ccpXHNfaoy.wav`, 360 s, en→de, chunk_ms=450): **BLEU 27.5, chrF 63.5, COMET 0.861, LongYAAL CU 1766 ms, LongYAAL CA 1473 ms, RTF 0.401.** Those runs were produced on an earlier, richer tuning surface; the manifest inside each output directory is the source of truth for exact runtime knobs. Full calibration curve and notes in [`docs/RESULTS.md`](docs/RESULTS.md).

## Docs map

- [`PLAN.md`](PLAN.md) — current plan and next steps
- [`DECISIONS.md`](DECISIONS.md) — append-only log of session-level decisions and what changed why
- [`docs/RUNTIME_ARCHITECTURE.md`](docs/RUNTIME_ARCHITECTURE.md) — ASR/MT axes, module map, session lifecycle
- [`docs/MT_VLLM_BACKEND.md`](docs/MT_VLLM_BACKEND.md) — design of the experimental `gemma_vllm_alignatt` MT backend (Phases 0–5)
- [`docs/CASCADE_POLICY_AUDIT.md`](docs/CASCADE_POLICY_AUDIT.md) — rationale for the shipped ASR->MT contract vs `baseline.py`
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
| `run_mt_backend_parity.py` | Isolated single-backend MT probe harness (historical name retained). |
| `run_context_ablation.py` | Three-condition MT paper-context ablation on one clip (hot bundle). |
| `evaluate_cascade_outputs.py` | BLEU / chrF / XCOMET-XL / LongYAAL over a run bundle. |

All other one-off research scripts live under [`scripts/`](scripts/).

## Extra-context (IWSLT 2026 sub-track)

ACL-paper PDFs can be preprocessed into compact `PaperArtifact` JSON and injected into the Gemma MT prompt as a `[Paper context]` block (retrieved top-k chunks via BM25 or static title+abstract). The runtime pairs with a **provenance guard** (`translation_alignatt_min_source_mass`) that vetoes any drafted token whose MT attention concentrates on the paper-context span, eliminating paper-content leakage observed with naive context injection.

Design, empirical ablations, and the submission-defensible setting live in [`docs/CONTEXT_INJECTION.md`](docs/CONTEXT_INJECTION.md). Default is `paper_context_mode=off` — extra-context injection is fully opt-in.

## Environments

- `.venv-inference` — all inference and streaming paths
- `.venv-evaluation` — OmniSTEval + XCOMET-XL (do not mix with the inference env)

## Validation

Live validation goes through the streaming runners and artifact evaluation:

```bash
.venv-inference/bin/python run_iwslt_submission.py batch \
  --preset main_low_latency \
  --source en \
  --target de \
  --input-dir dev-set/audio \
  --output-dir outputs/iwslt26_main_low_ende

.venv-evaluation/bin/python evaluate_cascade_outputs.py \
  --output-dir outputs/iwslt26_main_low_ende
```
