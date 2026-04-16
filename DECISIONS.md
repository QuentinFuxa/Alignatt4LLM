# DECISIONS.md — session log of 2026-04-16

This doc is a short, navigable log of what was decided during the
2026-04-16 session. It complements `PLAN.md` (which is the long-term
strategic direction), so a future agent can quickly understand *what
changed today and why* without rereading the whole PLAN.

For code-level rationale, see also the three commit messages from this
session (`git log --grep "AlignAtt-frontier\|explicit-language\|Merge branch 'full_vllm'"`).

## Starting state

- PLAN.md's "Critical Review (2026-04-16)" flagged that the streaming
  prefix-prefill claim (`RTF 1.387` on smoke18) was not a clean A/B and
  that the cascade needed "stability-based commit" to stop depending on
  model-emitted punctuation.
- On `tmp/alignatt_smoke18.wav` (A100 reference numbers from PLAN):
  `qwen_forced` RTF `0.798`, `gemma_onepass_qk_fast` RTF `2.950`,
  `gemma_vllm_qk_fast` RTF `2.305`, streaming variant crashes on
  longer clips.
- `full_vllm` branch existed in parallel: `mt_backend_name` runtime
  axis plus a full Gemma-MT-on-vLLM backend (observer/worker/tests).

## Key decisions

### 1. Ran PLAN step 1's same-SHA A/B on the ASR streaming claim

Three configurations on smoke18 (A40, same SHA):

| Config                                    | RTF   | Wallclock | Updates | Note                      |
|-------------------------------------------|-------|-----------|---------|---------------------------|
| `gemma_vllm_qk_fast` (`llm.chat`)         | 4.656 | 83.8 s    | 18      | baseline                  |
| `gemma_vllm_qk_fast` + `force_generate_api` (`llm.generate`, no prefix) | 4.706 | 84.7 s | 23 | API-path only |
| `gemma_vllm_qk_fast` + streaming prefix   | CRASH | —         | —       | `max_model_len=1024` overflow |

**Conclusion:** the `llm.chat → llm.generate(prompt_token_ids, multi_modal_data)` switch alone contributes **~0 RTF gain** (within noise).
Any streaming benefit comes from prefix-prefill itself, not from the
input-path change. This retired the PLAN's `1.387` number from the
claim list.

Code artefact: `gemma_vllm_force_generate_api` runtime flag (ablation
only, not a production switch).

### 2. Replaced the punctuation commit rule with AlignAtt-frontier

**Rule today (default, `asr_commit_mode="alignatt_frontier"`):** commit
every contiguous prefix of words whose AlignAtt-aligned `end_time` is
at least `asr_alignatt_frontier_margin_ms` (default 500 ms) behind the
current audio frontier.

**Why:**

- Strictly symmetric to the MT-side accessibility rule. The cascade
  becomes mono-mechanism (AlignAtt-frontier on both sides).
- Model-agnostic: works for Qwen, Gemma-onepass, Gemma-vllm — all
  three already produce word timings.
- Retires the structural failure where a model that never emits
  sentence-terminal punctuation (Gemma-4 E4B on smoke18) causes
  `utt_timestamps` to never advance and the prompt to overflow
  `max_model_len`. The streaming branch that crashed in point 1 now
  runs to completion.
- Interprets the margin as a latency buffer — the only knob is the
  latency the user is willing to pay.

Legacy rule kept as `asr_commit_mode="punctuation_lcp"` for ablation.

### 3. Empirical confirmation on smoke18 (same SHA, same machine)

| Config                                       | Commit rule        | RTF   | Updates | Prediction     |
|----------------------------------------------|--------------------|-------|---------|----------------|
| Gemma non-stream                             | punctuation_lcp    | 4.656 | 18      | **empty**      |
| Gemma stream                                 | punctuation_lcp    | CRASH | —       | —              |
| Gemma non-stream                             | alignatt_frontier  | 2.775 | 31      | non-empty      |
| Gemma stream                                 | alignatt_frontier  | 3.179 | 33      | non-empty      |
| Qwen non-stream (baseline)                   | punctuation_lcp    | 1.468 | 18      | clean German   |
| Qwen non-stream                              | alignatt_frontier  | 1.720 | 20      | earlier first emission (3.15 s vs 4.05 s), slightly fragmented MT |

On Gemma the RTF drops by ~40 % and the cascade actually produces
output. On Qwen the commit rule trades ~17 % RTF for ~900 ms of
first-emission latency — classic simultaneous tradeoff. The margin is
the lever; paper should include a sweep.

### 4. Gemma ASR prompt: switched to Google's documented phrasing

The previous prompt said "Transcribe the following speech segment in
its original language." Under streaming prefix-prefill the model
drifted into re-emitting the instruction itself
("Sicher, transkribiere das folgende Sprachsegment…"). Rewriting the
instruction as **"Transcribe the following speech segment in
{language} into {language} text."** (verbatim from Google's Gemma 4
audio guidance) eliminates the leak.

Also threaded the `language` argument all the way from
`CascadeRuntimeConfig.source_lang` through to the rendered prompt —
previously the backend accepted `language` and immediately
`del`-ed it.

### 5. Investigated the "prefix caching ↔ observer incompatibility"

`PLAN.md` section 8.5 described prefix caching making the decoded
text drift between cold and hot runs even after a host-side
prompt-observer cache restored observer completeness. Reading the
existing repeat artifacts side-by-side showed something different:

| enable_prefix_caching | cudagraph | run 0 (cold) | runs 1+ (hot) |
|-----------------------|-----------|--------------|----------------|
| False                 | full      | 44 tok, garbled | same           |
| True                  | full      | 44 tok, garbled | 53 tok, *punctuated, correct paper title* |

The hot text under prefix caching was **not** a drift artefact to
avoid — it was simply different decode numerics. Added a
`warmup(duration_seconds)` hook on the vLLM backend and re-ran the
repeat pattern. With an 18 s noise warmup pass, the cold/hot gap
collapses to a 1-token trailing difference on smoke18, and the
apparent "prefix-caching produces better text" observation
disappears — it was a cudagraph first-capture artefact, not a
semantic signal.

**Implication for PLAN.md:** the prefix-cache observer incompatibility
described in section 8.5 is a cudagraph capture artefact, not a
prefix-caching artefact. Observer completeness via the host-side
cache works as intended; text stability requires either warmup or
eager mode. Not a blocker for enabling prefix caching.

### 6. Merged `full_vllm`

Single commit `5f5119e` adds `mt_backend_name` as an independent
runtime axis, plus a Gemma-MT-on-vLLM backend (worker/observer) and a
parity harness. Merged into the branch containing today's ASR work
with three trivially co-resolved conflicts — the two efforts touch
the same files but disjoint regions.

## What stays open

- End-of-audio flush for `alignatt_frontier` is not special-cased: the
  final N words before EOS are never "margin behind the frontier" and
  so don't commit. The Qwen run's prediction ends mid-word
  ("durch Folgende Schritt für…"). A final chunk should commit
  everything remaining — small follow-up.
- Margin sweep (0 / 250 / 500 / 1000 / 2000 ms) on Qwen to
  characterise the latency↔quality curve. Good paper figure.
- Second-clip validation of `alignatt_frontier`
  (`test-set/audio/ccpXHNfaoy_short60s.wav` from PLAN), ideally on the
  A100 so absolute numbers are comparable to PLAN's reference table.
- Gemma-ASR intrinsic quality on smoke18 remains weak: hallucinations
  ("Samir" for "Si Yuan", "Funai" for "Fudan") and regurgitation of
  Google's `journal1.wav` training example ("Die Wettervorhersage
  prognostiziert morgen einen Höchstwert…"). This is a property of
  Gemma-4 E4B as ASR, not of the cascade infrastructure. Matches the
  PLAN's strategic pivot: keep `qwen_forced` on the ASR side, put
  the vLLM experimentation on the MT side.

## Where today's work fits in the paper story

- **Systems contribution** (already solid from earlier weeks):
  engine-native AlignAtt observer under vLLM `cudagraph=full`, compact
  per-token contract, `worker_cls` + on-device tensor observer.
- **Conceptual contribution added today:** AlignAtt-frontier commit as
  a mono-mechanism for both sides of the cascade, replacing the
  punctuation-dependent rule. Model-agnostic, empirically unblocks
  Gemma, documented latency↔quality tradeoff on Qwen.
- **Honest negative result:** Gemma-4 E4B as a standalone ASR model
  is worse than Qwen3-ASR-1.7B on this clip both in quality and in
  speed, for model-intrinsic reasons (size, ASR-specialisation). The
  strategic pivot to "Qwen ASR + Gemma MT through vLLM" (see updated
  PLAN.md) is supported by today's evidence.

---

## 2026-04-16 (late) — MT vLLM Phases 0–6 push + repo reorganization

### Phases 0–5 delivered

The `full_vllm`-merged `mt_backend_name` axis got fully fleshed out
with PLAN's Phase 0 → Phase 5 sequence:

- **Phase 0 (surface):** `VALID_MT_BACKEND_NAMES`,
  `STABLE_MT_BACKEND_NAMES`, `CascadeRuntimeConfig.mt_backend_name` (default stable),
  `build_mt_backend()` dispatcher, `--mt-backend-name` CLI flag, `_bundle_key`
  includes it, `LoadedModelBundle.ensure_mt_backend()` rebuilds on change.
  All runtime tests pass with no default behaviour change.
- **Phase 1:** minimal `gemma_vllm_mt_backend.py` — render same prompt as
  Transformers, `llm.generate(prompt_token_ids=...)`, deterministic decode.
  Three subtle fixes: trailing EOS token trimming, `mt_vllm_gpu_memory_utilization`
  bumped from 0.3 → 0.5 (Gemma-4 E4B weights are 15.28 GiB), subprocess
  isolation in the parity harness (cross-allocator GPU memory contamination
  between TF and vLLM in one process).
- **Phase 2:** engine-native MT observer via `gemma_vllm_mt_observer.py` +
  `gemma_vllm_mt_worker.py`. Parallel to the ASR-side observer but captures
  **K at prompt *and* decode positions** so the 4-way provenance partition
  (accessible / inaccessible / non-source / suffix) can be reconstructed
  from `softmax([prompt_K | decode_K])`. Single-prompt validation:
  both backends agree on blocked frontier and stop reason on two partial
  prompts.
- **Phase 3:** policy loop integrated; observer sequence trimmed to the
  draft length (drop trailing EOS), same stop-reason vocabulary
  (`alignatt:source_frontier` / `rewind` / `provenance_weak`). Curated 6-prompt
  parity: draft text matches 6/6, stop reason and blocked frontier match 5/6.
  Provenance *magnitudes* diverge because of numerical drift between vLLM's
  fused-QKV + proportional-RoPE path and Transformers' separate projections —
  documented in `docs/MT_VLLM_BACKEND.md` as expected-and-known.
- **Phase 4:** `run_mt_backend_parity.py` extended to a `--prompt-set`
  curated harness; each backend runs in its own subprocess (avoids
  the cross-allocator issue above).
- **Phase 5:** end-to-end SimulStream with `qwen_forced` + `gemma_vllm_alignatt`
  on `tmp/alignatt_smoke18.wav` — RTF 0.536, coherent German, no crashes,
  no observer failures. No runtime code change beyond Phase 0's surface work.

### Phase 6 (measurement, in progress)

Single-clip numbers on `test-set/audio/ccpXHNfaoy.wav` (360 s, en→de) at
baseline latency (`chunk_ms=450`):

| BLEU | chrF | COMET-XL | LongYAAL CU | LongYAAL CA | RTF |
|------|------|----------|-------------|-------------|-----|
| 27.51 | 63.54 | 0.861 | 1766 ms | 1473 ms | 0.401 |

Chunk-size calibration curve on `OiqEWDVtWk.wav` (299 s, en→de):

| chunk_ms | CA | CU | BLEU | COMET |
|---|---|---|---|---|
| 450 | 1650 | 1931 | 26.96 | 0.830 |
| **700** | **3539** | **3789** | **31.25** | **0.889** |
| 850 | 4740 | 5024 | 36.95 | 0.914 |
| 1500 | 7169 | 7556 | 38.91 | 0.924 |

Full results and caveats in `docs/RESULTS.md`.

**Key empirical finding:** `translation_alignatt_inaccessible_ms` has
effectively **zero** effect on LongYAAL CA in this architecture (tested
at 0 vs 2000 ms on the same clip: both 1650 ms CA, identical BLEU).
The cascade's scheduler already waits on commits/finals, so the
partial-only accessibility mask doesn't bite. **Chunk size is the clean
latency knob.**

### Repo reorganization

Pre-reorg: 47 `.py` files and 6 `.md` files at the repo root; big design
doc at `assets/alignatt_doc/E4B_ALIGNATT_CASCADE_DESIGN.md` (~1700 lines)
mixed current and historical content.

Post-reorg:

- **30 `.py` files at root**, all actively maintained. 16 dated research
  scripts moved to `scripts/` (with a `scripts/README.md` explaining
  they're preserved for reference only and how to run them via
  `PYTHONPATH=.`).
- **Docs centralized under `docs/`:**
  - `docs/RUNTIME_ARCHITECTURE.md` (new) — current ASR + MT axes, module map
  - `docs/MT_VLLM_BACKEND.md` (new) — Phase 0–5 design incl. Gemma-4 architecture quirks
  - `docs/RESULTS.md` (new) — consolidated quality/latency numbers
  - `docs/TROUBLESHOOTING.md` (new) — GPU gotchas (compile cache, utilization, cross-allocator)
  - `docs/archive/` — `E4B_ALIGNATT_CASCADE_DESIGN.md`, `ALIGNATT_LLM.md`,
    `SIMULSTREAM_TWO_FRONTENDS.md`, `PLAN_test_cs_en_qwen3_gemma.md`,
    the pre-reorg `PLAN_HISTORY_2026-04.md`, and the existing archive
  - `docs/reference/` — upstream `Qwen3_aligner.md` model card + AlignAtt
    reference paper/code (previously under `assets/alignatt_doc/`)
  - `assets/alignatt_doc/` removed (was mostly historical + upstream refs)
- **PLAN.md simplified** from 1682 lines to ~80. Historical narrative in
  `docs/archive/PLAN_HISTORY_2026-04.md`; current plan references
  `docs/RESULTS.md` / `docs/MT_VLLM_BACKEND.md` / `docs/TROUBLESHOOTING.md`
  for details.
- **`README.md`** added at root with entry-point map.
- **`AGENTS.md`** updated: acknowledges vLLM-on-both-sides, separate ASR/MT
  axes, adds a "where to find things" pointer section.

Full test suite passes after reorg (111/111). All active imports resolve.
All moved scripts still compile.

### What the repo looks like now (high level)

```
/
├── README.md                           # entry point
├── AGENTS.md, CLAUDE.md                # operational guidance
├── PLAN.md                             # short current plan
├── DECISIONS.md                        # append-only session log (this file)
├── docs/
│   ├── RUNTIME_ARCHITECTURE.md
│   ├── MT_VLLM_BACKEND.md
│   ├── RESULTS.md
│   ├── TROUBLESHOOTING.md
│   ├── archive/          # historical plans + design notes
│   └── reference/        # upstream model cards + reference code
├── scripts/              # dated research scripts (not maintained)
├── test_*.py             # pytest suite (at root)
├── run_simulstream_batch.py            # canonical runner
├── run_simulstream_compare.py
├── run_alignment_single_audio.py
├── run_mt_backend_parity.py
├── evaluate_cascade_outputs.py
├── cascade_*.py          # active runtime
├── alignment_backend.py, qwen_alignment_backend.py
├── gemma_alignment_probe.py
├── gemma_vllm_alignment_backend.py, gemma_vllm_worker.py       # Gemma ASR vLLM
└── gemma_vllm_mt_backend.py, gemma_vllm_mt_observer.py,
    gemma_vllm_mt_worker.py                                      # Gemma MT vLLM
```

### Open threads for the next session

See `PLAN.md` "Immediate next steps":

1. Multi-clip measurement on the 20-clip English test-set at two operating points.
2. Multilingual generalisation (en→it, en→zh) — heads and references already exist.
3. End-of-audio flush bug for `alignatt_frontier` (last N words never commit).
4. `asr_alignatt_frontier_margin_ms` sweep on Qwen for a paper figure.
5. Reopen MT vLLM prefix caching behind a cache-native observer port.
6. Investigate Phase 2/3 numerical drift on provenance magnitudes
   (argmax agrees, sums don't).
