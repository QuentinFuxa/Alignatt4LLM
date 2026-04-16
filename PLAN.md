# PLAN.md

Living plan for the simultaneous speech-translation cascade. Short and
focused; historical narrative has been archived to
[`docs/archive/PLAN_HISTORY_2026-04.md`](docs/archive/PLAN_HISTORY_2026-04.md).

For session-by-session decision log see [`DECISIONS.md`](DECISIONS.md).
For the runtime surface see [`docs/RUNTIME_ARCHITECTURE.md`](docs/RUNTIME_ARCHITECTURE.md).
For concrete numbers see [`docs/RESULTS.md`](docs/RESULTS.md).

## Primary direction

Ship, measure, and write up:

- ASR side: `qwen_forced` (Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B) via vLLM.
- MT side: `gemma_vllm_alignatt` (Gemma-4-E4B MT) via vLLM with an engine-native MT AlignAtt observer.
- Commit rule on both sides: **AlignAtt-frontier** (mono-mechanism).

Gemma ASR is reachable via `gemma_onepass_qk_fast` or `gemma_vllm_qk_fast` and works end-to-end, but Gemma-4-E4B as a standalone ASR model is intrinsically weaker than Qwen3-ASR-1.7B on our clips (hallucinations, regurgitation of training examples). This is a model-intrinsic property, not a cascade-infrastructure issue, so Gemma ASR stays as an experimental option rather than the default.

## Status snapshot (end of 2026-04-16 session)

Phases 0–5 of the "move Gemma MT from Transformers to vLLM" plan are delivered and end-to-end validated on one test-set clip:

| Phase | What it delivered | Status |
|---|---|---|
| 0 | `mt_backend_name` as an independent runtime axis + CLI surface + dispatcher + runtime defaults + tests | ✅ |
| 1 | Minimal `gemma_vllm_mt_backend.py` doing draft generation via vLLM | ✅ |
| 2 | `gemma_vllm_mt_observer.py` + `gemma_vllm_mt_worker.py` — engine-native MT AlignAtt observer with prompt-K, decode-Q, decode-K capture and 4-way provenance reconstruction | ✅ |
| 3 | Policy loop integrated on the vLLM side; same stop-reason vocabulary (`alignatt:source_frontier` / `rewind` / `provenance_weak`); curated 6-prompt parity; observer sequence trimmed to draft length | ✅ (decisions match on 5/6; numerical provenance drift documented) |
| 4 | Single-prompt MT parity harness with subprocess isolation per backend | ✅ |
| 5 | End-to-end SimulStream with `qwen_forced` + `gemma_vllm_alignatt` on `tmp/alignatt_smoke18.wav` | ✅ (RTF 0.536, coherent German, no observer failures) |

Phase 6 (measurement) is in progress — one-clip numbers on `ccpXHNfaoy.wav` and a chunk-size calibration curve on `OiqEWDVtWk.wav` are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Immediate next steps

1. **Multi-clip measurement** on the 20-clip English test-set at the two operating points:
   - `chunk_ms = 450` (sub-2 s LongYAAL CU regime)
   - `chunk_ms = 700` (sub-4 s LongYAAL CA regime — calibrated on `OiqEWDVtWk.wav`)
   Expected runtime ~1 hour each at RTF ~0.5. Write results into `docs/RESULTS.md`.

2. **Multilingual generalisation:** repeat the two operating points for en→it and en→zh. Head files and references already exist; the runtime auto-resolves via `alignatt_heads_path_for(source, target)`. en→cs is partially blocked (needs `test-set/ref/cs.txt`).

3. **End-of-audio flush** for `alignatt_frontier`: the final N words before EOS never satisfy "margin behind the frontier" and so never commit. A final chunk should commit everything remaining. Short follow-up; see `DECISIONS.md` for context.

4. **Margin sweep on Qwen ASR** (`asr_alignatt_frontier_margin_ms ∈ {0, 250, 500, 1000, 2000}`) on one clip to characterise the latency-quality curve of the commit rule itself. Good paper figure.

5. **Reopen prefix caching for MT vLLM.** Currently `mt_vllm_enable_prefix_caching=False`. The ASR-side host-side prompt-observer cache should be ported and keyed to the MT observer identity. Before this lands, add a vLLM compile-cache invalidation hook so switching between ASR-observer and MT-observer variants in the same environment doesn't require a manual `rm -rf ~/.cache/vllm/torch_compile_cache` (see `docs/TROUBLESHOOTING.md`).

6. **Phase 2/3 numerical drift investigation.** Acceptance decisions agree, but provenance *magnitudes* drift between vLLM's fused-QKV + proportional-RoPE path and Transformers' separate-projection path. Not a blocker (argmax is robust, downstream policy works), but would be nice to understand before writing up provenance-based claims.

## Hard rules

- Do **not** silently make `gemma_vllm_alignatt` the MT default. Keep `STABLE_MT_BACKEND_NAMES = ("gemma_transformers_alignatt",)`.
- Do **not** re-enable MT vLLM prefix caching without the cache-native observer port.
- Do **not** widen to a full benchmark sweep before the two-clip sanity check is clean.
- Do **not** conflate ASR-side and MT-side observer work. They are two separate substrates that happen to share a design pattern.
- Do **not** revive Gemma ASR fine-tuning inside this repo. The pivot is: keep Qwen ASR, put vLLM experimentation on the MT side.

## Paper-level framing (target claim)

*A multimodal causal LLM with an **ASR-side AlignAtt observer** (audio K + decode Q, `cudagraph=full`) and a matching **MT-side AlignAtt observer** (prompt K + decode Q + decode K, engine-native 4-way provenance), both based on a compact per-token contract, survives real vLLM execution and plugs into a single-process simultaneous speech translator whose emission on both sides is governed by the same AlignAtt-frontier rule. Runs at sub-2 s LongYAAL CU on `ccpXHNfaoy.wav` en→de, with a clean chunk-size calibration curve up to >4 s CA where BLEU / COMET saturate.*
