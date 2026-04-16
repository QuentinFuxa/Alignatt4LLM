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

---

## 2026-04-16 (night) — IWSLT prep: EOS flush + commit-rule default rollback

### EOS flush for end-of-stream

Both ASR commit rules have a failure mode at end-of-stream:

- `alignatt_frontier`: the last ~500 ms of words sit past the
  `asr_alignatt_frontier_margin_ms` gate and never commit.
- `punctuation_lcp`: if the last sentence has no terminal period,
  nothing after the last observed period commits.

`CascadeSession.finalize_stream` now calls `transcribe_audio(is_final_chunk=True)`
which threads through to both commit helpers. On a final chunk both helpers
flush the whole current hypothesis (alignatt_frontier: commit every aligned
word ignoring margin and LCP; punctuation_lcp: commit the whole hypothesis
regardless of terminal punctuation). Eight new unit tests in
`test_eos_flush.py` pin the fix without needing a GPU. On-clip effect is
small when the speaker ends with a natural pause (`ccpXHNfaoy.wav`:
732 → 733 words, +0.09 BLEU under alignatt_frontier) but net positive.

### Rolled back `asr_commit_mode` default from `alignatt_frontier` to `punctuation_lcp`

While validating the EOS fix on `ccpXHNfaoy.wav`, ran a same-SHA A/B
with only the commit rule changed. Measured en→de on the canonical clip:

| Commit rule | BLEU | chrF | COMET | LongYAAL CU | LongYAAL CA | RTF |
|---|---|---|---|---|---|---|
| `punctuation_lcp` | **27.51** | **63.54** | **0.861** | 1766 | 1483 | 0.42 |
| `alignatt_frontier` | 15.78 | 54.43 | 0.558 | 1328 | 1048 | 0.43 |

AlignAtt-frontier buys ~440 ms LongYAAL CA at the cost of
**−11.7 BLEU / −9 chrF / −0.30 COMET**. That ratio is not worth it for
any Qwen-ASR submission path. The earlier switch to `alignatt_frontier`
(morning session, commit `81c0376`) was correct for the Gemma-ASR path
— Gemma-4-E4B doesn't emit sentence-terminal punctuation on smoke18, so
`utt_timestamps` never advanced — but it's catastrophic for Qwen, whose
clean punctuation output gives MT complete sentences to translate.

Rolled back the global default; AlignAtt-frontier is now opt-in via
`--asr-commit-mode alignatt_frontier`. The pivot in `PLAN.md` (Qwen ASR
as default, Gemma ASR experimental) makes this a clean swap: default
submission path regains its pre-rollout quality, Gemma-ASR still works
by explicit opt-in.

Test updates: `test_alignment_helpers.py` assertions flipped accordingly;
new test confirms `alignatt_frontier` remains accepted as an opt-in
value.

---

## 2026-04-16 (overnight) — Step 1 submission hardening + Step 2 re-anchor + Step 4 mechanism branch

### Step 1 delivered: three submission-hardening fixes

All three config-only, regression-tested without loading any model.
Details in commit `16609ec`.

1. **`LANGUAGE_CODE_TO_NAME` covers Czech.** Duplicate
   `LANGUAGE_NAME_TO_CODE` declarations (one before Czech, one after)
   left the reverse map built from the pre-Czech version, so `cs` would
   silently pass through as the raw code instead of `"Czech"` anywhere
   downstream read back a human-readable label. Consolidated to one
   definition; reverse map derived from it.

2. **Heads-path refresh on either language change.** `apply_overrides`
   and `temporary_runtime_config` now recompute
   `translation_alignatt_heads_path` when either `source_lang` or
   `target_lang` changes. Previously only `target_lang` triggered the
   refresh, so `cs->en` kept the English-source heads file.

3. **Bundle fingerprints replace bare backend-name identity.**
   `LoadedModelBundle` now tracks `alignment_backend_fingerprint()`
   and `mt_backend_fingerprint()`: the tuple of engine-construction
   knobs (gpu_memory_utilization, prefix_caching, cudagraph_mode,
   max_model_len, prompt_kv_reuse, ...). Flipping any of those under
   hot reuse now triggers a rebuild; flipping live policy (commit
   mode, heads path, thresholds, language) reuses the hot backend.
   The test pins both directions of the split.

Full suite: 124/124 pass.

### Step 2 delivered: canonical baseline re-anchored at two operating points

Pair: `qwen_forced` ASR + `gemma_vllm_alignatt` MT, `punctuation_lcp`,
`translation_alignatt_heads_path` = `en-de`. Run on
`test-set/audio/ccpXHNfaoy.wav` (360 s) in a single driver invocation
(hot model reuse across chunk sizes).

| Operating point | BLEU  | chrF  | LongYAAL CU | LongYAAL CA | RTF   |
|-----------------|-------|-------|-------------|-------------|-------|
| chunk_ms=450    | 27.51 | 63.54 | 1766 ms     | 1466 ms     | 0.393 |
| chunk_ms=700    | 38.19 | 66.53 | 3275 ms     | 2945 ms     | 0.369 |

chunk_ms=450 numerics are **identical** to the pre-hardening
`simulstream_phase6_one_clip` run (BLEU 27.5133, chrF 63.5404, CU
1766.3516) — confirms the Step 1 changes are truly config-only.

chunk_ms=700 buys +10.7 BLEU / +3.0 chrF for +1509 ms CU / +1479 ms CA
on this clip. A clean two-operating-point story for the paper.

### Step 4 delivered (code-only): `stable_and_accessible` commit rule

New third ASR commit mode on top of `punctuation_lcp` and
`alignatt_frontier`. A word is committable iff it is **both**:

- *accessible*: aligned end_time is at least
  `asr_alignatt_frontier_margin_ms` behind the audio frontier (same
  rule as `alignatt_frontier`);
- *stable*: identical at the same position in the last K consecutive
  ASR hypotheses for the current utterance segment, with
  `asr_stability_k` controlling K (default 3, must be ≥ 2).

K=2 is provably equivalent to `alignatt_frontier` (pinned by test
`test_stable_and_accessible_k2_matches_alignatt_frontier_behavior`).
K≥3 is strictly more conservative and costs K-1 extra chunks of
buffering before the first commit in each utterance segment.

Design intent (paper framing):

- Generalises the current `alignatt_frontier` rule from its weakest
  (K=2) stability signal to arbitrary K.
- Collapses the "how do we commit a source unit" question to a
  single 2D hyperparameter surface (margin M, stability K) rather
  than a dispatch over punctuation-dependent vs frontier-dependent
  modes.
- Model-agnostic (works for Qwen3-ASR or Gemma-4 ASR), so
  `punctuation_lcp` no longer needs to stay as a model-conditional
  fallback.

Still pending: on-clip measurement (GPU run in flight) to verify
that K=3 closes the BLEU gap vs `punctuation_lcp` (27.51) without
eating the CA win vs pure punctuation-gating.

### Status (end of overnight)

- **Step 1:** completed (`16609ec`).
- **Step 2:** completed (chunk_ms=450 and chunk_ms=700 on `ccpXHNfaoy.wav`).
- **Step 3:** completed — en→de, en→it, en→zh all produce coherent
  output under the hardened runtime, no direction-specific breakage.
- **Step 4:** code + tests landed in `7ab5a39` + sentinel-fix `7d27eec`;
  K=3 and K=4 measured on `ccpXHNfaoy.wav`. See "mechanism-branch
  findings" below. Honest negative result on the Qwen-ASR path: the
  rule is a strict improvement over `alignatt_frontier` but is
  Pareto-dominated by `punctuation_lcp` on models that emit clean
  punctuation. Paper framing updated accordingly.
- **Step 5 (fallback MT prefix caching):** skipped. The main mechanism
  branch produced clean evidence and a defensible paper result, so
  the "fallback-only if main is a dead end" gate does not fire.
- **Step 6 (cheap follow-ups):** completed. See "Step 6 findings" below.
- **Step 7 (stretch/paper branches):** unblocked. `run_simulstream_batch`
  now emits `alignatt_metadata`, `partial_accepted_target`,
  `partial_draft_target`, `asr_text`, and MT prompt-token counters
  per stream update (commit `a0edcc6`). The night1 K=3@700 artifact
  is the first dataset with full observer metadata attached, usable
  for future offline continuous-confidence replay without re-running
  the GPU pipeline. The actual replay pass was not run tonight
  (needs a dedicated replay driver) but the prior blocker no longer
  applies.

### Mechanism-branch findings: stable_and_accessible K-sweep

Same clip (`ccpXHNfaoy.wav`), same configuration except ASR commit rule.

| Commit rule                    | BLEU  | chrF  | COMET | LongYAAL CU | LongYAAL CA |
|--------------------------------|-------|-------|-------|-------------|-------------|
| `alignatt_frontier`  (K=2)     | 15.78 | 54.43 | 0.558 | 1328 ms     | 1048 ms     |
| `stable_and_accessible` K=3    | 18.71 | 56.37 | 0.681 | 1919 ms     | 1637 ms     |
| `stable_and_accessible` K=4    | 20.26 | 57.92 | 0.730 | 2510 ms     | 2240 ms     |
| `stable_and_accessible` K=5    | 25.79 | 60.91 | 0.782 | 3585 ms     | 3395 ms     |
| `stable_and_accessible` K=6    | 28.13 | 62.16 | 0.824 | 4231 ms     | 4204 ms     |
| `punctuation_lcp`              | 27.51 | 63.54 | 0.861 | 1766 ms     | 1466 ms     |

K=5 and K=6 added in the late-night extension of the sweep. K growth
is not linear: the K=4→K=5 step delivered +5.53 BLEU (a phase
transition where tail-word flicker stops dominating), after which
K=5→K=6 added +2.34 BLEU (saturation). By K=6 the frontier rule
actually **matches or narrowly exceeds punct on BLEU** (28.13 vs
27.51) but still loses on chrF and COMET, and pays a ~2.7 s CA
penalty for the privilege (4204 ms vs 1466 ms). `punctuation_lcp`
stays Pareto-optimal: no K value dominates it on more than one of
{BLEU, chrF, COMET, CA} at once.

**Cross-latency check:** running the same rule at chunk_ms=700:

| Commit rule                    | chunk_ms | BLEU  | chrF  | COMET | CU     | CA     |
|--------------------------------|----------|-------|-------|-------|--------|--------|
| `stable_and_accessible` K=3    | 450      | 18.71 | 56.37 | 0.681 | 1919 ms| 1637 ms|
| `stable_and_accessible` K=3    | **700**  | 24.67 | 60.12 | 0.740 | 2829 ms| 2521 ms|
| `punctuation_lcp`              | 450      | 27.51 | 63.54 | 0.861 | 1766 ms| 1466 ms|
| `punctuation_lcp`              | 700      | 38.19 | 66.53 | 0.940 | 3275 ms| 2945 ms|

Longer chunks help the frontier family (K=3 gains +5.96 BLEU / +0.059
COMET) because each chunk delivers more source context per MT call and
per-commit fragments are smaller. But `punctuation_lcp` still
Pareto-dominates every frontier operating point at every chunk size.
The fragmentation penalty is the intrinsic cost of word-level source
commits; chunk size moderates it but does not eliminate it.

Why the rule underperforms on Qwen-ASR + Gemma MT: frontier-based
commits fragment MT context. Word-level commits force MT to emit
mid-sentence target fragments that compound into fluency degradation;
`punctuation_lcp` hands MT whole sentences and MT fluency stays intact.
The quality cliff of `alignatt_frontier` is not caused by unstable
ASR words (which K catches) but by how the cascade couples source-
commit choice to MT emission granularity.

**Practical outcome:**
- `punctuation_lcp` stays the Qwen-ASR submission default.
- `stable_and_accessible` replaces `alignatt_frontier` as the
  recommended model-agnostic fallback for ASR paths without reliable
  sentence punctuation. K is the exposed knob (default 3, validated
  ≥ 2).
- `alignatt_frontier` is retained as the K=2 equivalence class of
  `stable_and_accessible` — no behavioural change, just a cleaner
  story.

### Widening scores (all chunk_ms=450, `punctuation_lcp`, ccpXHNfaoy.wav)

| Direction | BLEU  | chrF  | COMET | LongYAAL CU | LongYAAL CA | RTF   |
|-----------|-------|-------|-------|-------------|-------------|-------|
| en → de   | 27.51 | 63.54 | 0.861 | 1766 ms     | 1466 ms     | 0.393 |
| en → it   | 37.75 | 71.81 | 0.770 | 1848 ms     | 1567 ms     | 0.400 |
| en → zh   | 42.33 | 38.37 | 0.739 | 1781 ms     | 1634 ms     | 0.375 |

No direction-specific runtime breakage on these three. Italian output
is qualitatively coherent.

cs→en runtime check on `csJIsDTYMW.wav` first attempted with the
canonical `gemma_vllm_alignatt` MT backend and hit a genuine runtime
bug: during vLLM's `_initialize_kv_caches` → `_dummy_run` memory
profiling pass, the AOT-compiled Gemma4 forward raises
`KeyError: '_alignatt_mt_qk_tensor_observer'`. The MT worker defers
observer install until `prepare_mt_observer` arms it, and the
profiling pass runs the forward before that — the patched forward
tolerates the missing observer via `getattr(..., default=None)`,
but the **AOT-compiled cached** forward appears to perform a direct
dict access that bypasses `getattr` and raises `KeyError` when the
attribute is absent. This is not a cs→en-specific problem; it is
a torch-compile-cache / observer-init ordering interaction that the
night's en→de runs happened to avoid because of cache warmth on
their specific input shapes. See `tmp/cs_en_runtime_check.log` for
the full trace.

**Fix landed later in the same session (commit `70c0492`):**
`install_stub_observers_on_model` seeds every Gemma4Attention
layer's `_alignatt_mt_qk_tensor_observer` attribute with an explicit
`None` at `load_model` time, before vLLM's memory profiling fires.
The existing `_get_mt_qk_tensor_observer` → `_capture_mt_qk_into_tensor_buffers`
pair already treats `None` as "no observer configured" and
early-returns, so pre-seeding puts the attribute in `__dict__`
without changing behaviour. Verified by rerunning cs→en with
`mt_backend_name="gemma_vllm_alignatt"` on the canonical path:
no KeyError, models loaded in 117.7 s, inference RTF 0.544 (vs
Transformers MT RTF 1.377 on the same clip — 2.5× speedup). The
output is bit-identical in head text between the Transformers-MT
workaround and the fixed vLLM-MT canonical path.

**Fragility caveat (discovered in a late-night retry):** the fix
works reliably on a fresh-compile pass (cache miss → recompile →
captures stubs correctly). On cache HIT, the interaction between
vLLM's AOT compile cache and torchinductor's cached Python-side
artefacts is still fragile: a subsequent run that tries to reuse
the compiled graph can surface either (a) the original KeyError
(when `__dict__` lookup on a fresh attention instance misses the
stub for subtle reasons) or (b) a secondary
`ValueError: too many values to unpack (expected 20)` coming from a
stale torchinductor file whose signature doesn't match the freshly
loaded vLLM AOT graph. Full cache clearing
(`rm -rf /home/.cache/vllm/torch_compile_cache/torch_aot_compile/
f5ee.../ && rm -rf /tmp/torchinductor_root`) between tries is
currently the only fully reliable workaround. The first run in a
session, especially after any edit to `gemma_vllm_mt_observer.py`
or `gemma_vllm_mt_worker.py`, lands cleanly; subsequent retries may
need a cache wipe. A robust fix would need to either (i) teach the
compiled Gemma4 forward to tolerate a missing observer attribute
(rather than stub-patching it in at worker init), or (ii) invalidate
the torchinductor cache consistently with the vLLM AOT cache. Both
are out of tonight's scope.

Re-run with `mt_backend_name="gemma_transformers_alignatt"` sidesteps
the observer/compile-cache issue and exercises the Step 1 language-map
+ heads-path fixes end-to-end. Later, with the stub-observer fix
shipped, the canonical vLLM-MT path runs cleanly on the same clip:

| Config                                | Audio dur | RTF   | Updates |
|---------------------------------------|-----------|-------|---------|
| cs→en Transformers-MT chunk_ms=450    | 352 s     | 1.377 | 444     |
| cs→en **vLLM-MT** chunk_ms=450 (fixed) | 352 s     | **0.544** | 473     |

Bit-identical head text between the two backends confirms the fix
preserves correctness.

Prediction head (first 300 chars):

> *"Hello, my name is Kaiyuan and I will be presenting our work
> titled 'When Does Translation Require Context?' a data-driven
> multilingual exploration. This work was based in collaboration with
> Patrick Ferdinand, Emil, and Andrej F. Martins and Graham Newbig.
> Yes. So many translations depend on context."*

Clean coherent English output from Czech speech. Step 1's language
map + heads-path fix validated end-to-end: `LANGUAGE_CODE_TO_NAME["cs"]
== "Czech"`, `alignatt_heads_path_for("Czech", "English")` resolves to
`translation_heads_google_gemma-4-E4B-it_cs-en.json`, cascade produces
structured output. No quantitative eval (local `test-set/ref/` has no
cs-en reference).

Artifact: `outputs/night1_cs_en_chunk450/` — also carries the new
per-update `alignatt_metadata` schema so it is usable for future
replay work.

### Step 7 (stretch): continuous-confidence offline replay

Landed as `scripts/continuous_confidence_replay.py`. Reads a
stream_updates.jsonl with observer metadata and derives a per-token
continuous confidence scalar from the 4-way provenance distribution
(`source_accessible`, `source_inaccessible`, `non_source_prompt`,
`suffix`):

    conf_raw = 0.5 * source_accessible
             - 0.2 * source_inaccessible
             - 0.3 * (entropy_nats / log(4))
             + 0.3
    conf    = sigmoid(6 * (conf_raw - 0.5))

Compared against the online runtime's discrete accept/reject decisions:

| Artifact                                   | accepted mean | rejected mean | F1 at best single threshold |
|--------------------------------------------|---------------|---------------|-----------------------------|
| `night1_ende_stable_k3_chunk700` (en→de)   | 0.225         | 0.217         | 0.717 @ thr=0.15            |
| `night1_cs_en_chunk450` (cs→en)            | 0.356         | 0.298         | 0.649 @ thr=0.20            |

**Honest result:** the naive linear scalar **does not cleanly replicate
the discrete-gate decisions**. Mean confidence deltas between accepted
and rejected tokens are small (≤ 0.06) and best-F1 plateaus around
0.65–0.72. Even per-stop-reason means cluster tightly
(e.g. `alignatt:source_frontier` mean 0.22 vs `alignatt:rewind` mean
0.25 on the en→de artifact — not separable).

Interpretation for the paper: the 4-way provenance distribution, by
itself, carries weak discriminative information about commit safety.
The discrete gates clearly use quantities beyond provenance magnitudes
— positional indices (`unsafe_target_token_index`, frontier positions),
accessibility counts, and rewind thresholds. A continuous scalar that
genuinely replaces the three-gate policy would need to incorporate
these, or would need per-head weights learned from labelled commit
decisions.

Artifacts: `outputs/night1_*/confidence_replay_report.txt` and
`confidence_replay.csv`. The CSV is per-token and carries the raw
provenance plus derived scalar, so follow-up replay experiments
(different weightings, richer features, learned classifiers) can
run against it without re-scoring inference.

The existence of a running replay driver closes the Step 7 loop:
the schema instrumentation is usable, the first-pass analysis gives
a defensible negative result, and the data artefact is in place for
follow-up exploration.

### Step 7 follow-up: per-gate separability

The aggregate accept/reject F1 turned out to be the wrong framing.
The three discrete gates fire on different conditions, so pooling
them hides per-gate signal. Added `scripts/per_gate_separability.py`
(aggregates per-token rows into per-update records, then for each
discrete stop_reason searches over features × threshold × direction
for the best single-scalar predictor of "this gate fires on this
update").

Cross-artifact results, top feature per gate:

| Gate                       | Feature                               | Direction | F1 (en→de K3@700) | F1 (cs→en) |
|----------------------------|---------------------------------------|-----------|-------------------|------------|
| `alignatt:source_frontier` | `unsafe_token.source_inaccessible`    | ≥ 0.000   | **0.978**         | **0.910**  |
| `alignatt:rewind`          | `unsafe_token.source_inaccessible`    | ≤ 0.000   | 0.750             | 0.723      |

**Sharpened paper result:**

- **`alignatt:source_frontier` is cleanly absorbed by a continuous
  scalar**: a single threshold on the attention mass the unsafe-flagged
  draft token places on source-inaccessible positions reproduces the
  gate at F1 0.91-0.98 across both directions. This is the most
  load-bearing of the three gates, and it turns out to already be a
  discrete readout of a continuous quantity.
- **`alignatt:rewind` does not collapse**: under the same feature family
  it caps at F1 ~0.72-0.75. Rewind fires on the **absence** of
  source-inaccessible attention (opposite direction), but the
  separation is much weaker, consistent with rewind being a distinct
  mechanism that also consumes positional and threshold-based state
  (`translation_alignatt_rewind_threshold`) that isn't exposed in the
  provenance distribution alone.

**Paper framing this enables:** the three-gate policy is structurally
asymmetric. `source_frontier` is a reducible gate — the natural paper
move is to promote the continuous scalar as the primary mechanism and
recover `source_frontier` as a one-line threshold. `rewind` stays as
an independent mechanism that can later be studied on its own terms
(positional features, learned weights, etc). This is the right shape
of the contribution: one gate becomes continuous, one stays discrete,
with empirical support for why.

Artifacts: `outputs/night1_*/per_gate_separability.txt`. Source:
`scripts/per_gate_separability.py`.

### Step 7 v2: per-gate separability with positional / monotonicity features

The v1 analysis used only provenance features. Rewind is *defined*
by a backward jump in `aligned_source_local_positions`, so the v2
script (`scripts/per_gate_separability_v2.py`) reads
stream_updates.jsonl directly and adds:

- `max_backward_jump` — max(aligned[i] − aligned[i+1]) over adjacent draft tokens
- `n_backward_pairs`, `monotonicity_ratio`, `position_drift`
- `unsafe_idx_ratio`, `accepted_ratio`, `accessibility_ratio`
- `n_tokens`, `source_unit_count`, etc.

Cross-artifact top-feature table under v2 (n_pos / F1 per gate):

| Gate                       | Artifact            | Top feature (v2)                       | F1    |
|----------------------------|---------------------|----------------------------------------|-------|
| `alignatt:source_frontier` | en→de K3@700        | `unsafe_token.source_inaccessible`     | 0.978 |
| `alignatt:source_frontier` | cs→en               | `unsafe_token.source_inaccessible`     | 0.910 |
| `alignatt:rewind`          | en→de K3@700 (n=10) | `max_backward_jump ≥ 9`                | 0.667 |
| `alignatt:rewind`          | cs→en (n=64)        | `max_backward_jump ≥ 9`                | 0.696 |

**Positional features do not rescue rewind.** Even `max_backward_jump`,
which is the closest-to-definition feature for rewind, caps at F1
~0.67-0.70 across both artifacts — *lower* than the provenance-based
v1 cap of F1 0.72-0.75. This is the kind of confirmation the paper
needed: no single feature we've measured cleanly replicates the
rewind gate; the discrete gate carries state (threshold config,
accepted-window history, rewind_from/to positions) that a continuous
scalar cannot recover from observer provenance alone.

**Paper claim that survives tonight:** the three-gate MT policy
partitions cleanly into (i) **one reducible gate** — `source_frontier`
is a discrete readout of a continuous provenance quantity — and (ii)
**one irreducible gate** — `rewind` depends on state beyond
observer provenance. The continuous-confidence paper direction can
absorb (i) cleanly and must leave (ii) as a distinct mechanism.

Artifacts: `outputs/night1_*/per_gate_separability_v2.txt`.

### Step 7 v3: 2-feature search for the rewind gate

Added `scripts/two_feature_gate_search.py`, which grid-searches over
every pair of features (from the v2 feature set) with every
(direction_a, direction_b, AND/OR) combination rule, thresholds
sampled from per-feature quantiles. Asks whether any 2-feature
conjunction or disjunction lifts the rewind gate above the
single-feature cap.

Cross-artifact best 2-feature rule for `alignatt:rewind`:

| Artifact                        | n_pos | Best 2-feature rule (top F1)                                                 | F1    |
|---------------------------------|-------|------------------------------------------------------------------------------|-------|
| cs→en Transformers MT           |   64  | `max_backward_jump ≥ 10 AND unsafe.source_inaccessible ≤ 0.013`              | 0.734 |
| cs→en vLLM MT (fixed)           |   38  | `max_backward_jump ≥ 10 AND unsafe.source_inaccessible ≤ 0.008`              | 0.674 |
| en→de K3@700 (n_pos=10, noisy)  |   10  | `max_backward_jump ≥ 9 AND accepted_mean.source_accessible ≥ 0.044`          | 0.818 |

**Finding stands and sharpens:** 2-feature combinations do not
reliably lift `alignatt:rewind` on the two larger-sample artifacts
(n_pos=38 and 64). The en→de K3@700 case pushes higher (F1 0.818)
but only has 10 positive examples — that F1 is statistically fragile
and not a trustworthy lift. The best 2-feature combinations all
**agree on the same physical rule**:

> "a large backward jump in aligned source positions, **without**
> the attention-to-inaccessible-source that would instead trigger
> `source_frontier`."

That interpretation is satisfying — it's the **definition of what
rewind should fire on** — but the data shows even the well-motivated
combination caps at F1 0.67-0.73 on realistic sample sizes. The
remaining gap comes from state the observer does not expose per
update (`translation_alignatt_rewind_threshold` counter, accepted
window history across updates).

**Paper conclusion stands:** the three-gate MT policy partitions
into one *reducible* gate (`source_frontier`, F1 0.91-0.98 with one
provenance feature) and one *irreducible-under-observer-features*
gate (`rewind`, F1 plateau ≤ 0.75 across 1-feature and 2-feature
searches). Making `rewind` reducible would need either learned
per-head weights, a state-carrying classifier, or instrumenting the
rewind counter into `alignatt_metadata`.

Artifacts: `outputs/night1_*/two_feature_search_alignatt_rewind.txt`.
Source: `scripts/two_feature_gate_search.py`.

### Step 7 v4: loop-replay predictor — perfect gate recovery, definitive framing

The 1-feature / 2-feature searches told me *rewind* doesn't cleanly
reduce to a scalar. Before calling it "irreducible" I wanted one
more bounded check: run the MT policy's loop logic offline against
`aligned_source_local_positions` and `accessible_source_local_end_exclusive`
from the metadata.

`scripts/loop_replay_gate_predictor.py` mirrors
`cascade_mt_backend.policy.should_stop_in_loop`:

```
last_aligned = None
for current in aligned:
    if current is not None and last_aligned is not None:
        if last_aligned - current > rewind_threshold: return "alignatt:rewind"
    if current is not None and current >= accessible_end: return "alignatt:source_frontier"
    if current is not None: last_aligned = current
return "stop"
```

Cross-artifact, with `rewind_threshold=8`:

| Artifact                              | `alignatt:rewind` F1 | `alignatt:source_frontier` F1 |
|---------------------------------------|----------------------|-------------------------------|
| cs→en vLLM MT (fixed path, n=38/114)  | **1.000**            | **1.000**                     |
| cs→en Transformers MT (n=64/167)      | **1.000**            | **1.000**                     |
| en→de K3@700 (n=10/87)                | **1.000**            | **1.000**                     |

**Confusion matrices show zero cross-class errors for the two unsafe
gates.** `length` / `<turn|>` reasons (loop ran to completion) are
predicted as `stop`, which is correct for the rewind/source_frontier
binary question.

**Definitive paper framing on the continuous-confidence question:**

- The observer metadata **does** contain enough information to fully
  recover the three-gate policy — F1 is exactly 1.0 for both
  `source_frontier` and `rewind`.
- That recovery is **not a scalar threshold**. It's a loop replay:
  "which unsafe condition fires first in the token-by-token scan".
- Single-feature and 2-feature searches cap at F1 0.73-0.98 because
  multiple tokens per update may individually satisfy unsafe
  conditions, but only the **first firing** matters (the policy
  `break`s). Scalar features cannot express that sequential
  first-occurrence semantics.
- So the paper pitch "collapse discrete gates into a single
  continuous scalar" is **well-posed and well-measured negative**:
  the policy is a token-level sequential decision, not an
  update-level scalar judgment. What can be collapsed is the
  per-token *accessibility* test (that's the
  `unsafe.source_inaccessible` threshold that gave F1 0.91-0.98 on
  `source_frontier` — because for source_frontier, any firing token
  signals the gate). What cannot be collapsed is the interaction
  between multiple first-fire candidates within one update.

**Paper contribution this supports:**

> A provenance-only continuous confidence scalar absorbs
> `source_frontier` cleanly (one threshold, F1 0.91-0.98 across two
> language directions and two MT backends). The same scalar family
> does not absorb `rewind`, and we show the reason is structural:
> the three-gate policy is a first-unsafe-wins loop, not an
> update-level scalar classifier. The full policy is recoverable
> from the observer metadata via loop replay (F1 exactly 1.0), so
> the observer contract is complete in what it exposes — the
> "continuous scalar" question is specifically about what
> single-value thresholds can and cannot express about a loop-break
> decision.

Artifacts: `outputs/night1_*/loop_replay_gate_prediction.txt`.
Source: `scripts/loop_replay_gate_predictor.py`.

### Attempted: rerun canonical en→de baseline with instrumented schema

Goal was to regenerate `outputs/reanchor_chunk450` (canonical en→de
submission baseline at chunk_ms=450) with the new observer-
instrumented schema so loop-replay analysis could cover the
submission path itself. Ran three times, all failed at vLLM engine
init with the compile-cache fragility issue documented above — first
with the original KeyError, then twice with the secondary
`ValueError: too many values to unpack (expected 20)` coming from a
stale torchinductor file mismatch after repeated cache clears.

**Not critical for the paper conclusions:**

- The loop-replay F1 = 1.000 finding is already validated on three
  artifacts with the instrumented schema (`night1_cs_en_vllm_mt_chunk450`,
  `night1_cs_en_chunk450`, `night1_ende_stable_k3_chunk700`), spanning
  two language directions and two MT backends.
- The en→de BLEU / chrF / COMET / CA numbers are already measured on
  `reanchor_chunk450` (the pre-instrumentation artifact), and
  identical to the earlier `simulstream_phase6_one_clip` result
  bit-by-bit.
- The canonical path works end-to-end on fresh caches; the
  retry-fragility is a known workaround issue, not a blocker on a
  clean run.

Documented as a follow-up engineering task, not as a paper-level
limitation.

**Fallback path succeeded on 5th attempt:** ran the same config with
`mt_backend_name="gemma_transformers_alignatt"` (Transformers MT
instead of vLLM MT) — sidesteps the vLLM compile-cache fragility
entirely. Result on `ccpXHNfaoy.wav`:

| Metric          | Canonical Transformers MT (instrumented)    | vLLM MT re-anchor (pre-instrumentation)    |
|-----------------|---------------------------------------------|--------------------------------------------|
| BLEU            | 28.22                                       | 27.51                                      |
| chrF            | 63.53                                       | 63.54                                      |
| COMET           | 0.862                                       | 0.861                                      |
| LongYAAL CU     | 1747 ms                                     | 1766 ms                                    |
| LongYAAL CA     | 2240 ms                                     | 1466 ms                                    |
| RTF             | 1.020                                       | 0.393                                      |

Quality metrics match (BLEU +0.7, chrF identical, COMET +0.001,
within expected cross-backend drift). CA differs because Transformers
MT is 2.5× slower than vLLM MT — a wallclock-elapsed measurement,
not a semantic difference. The instrumented artifact carries full
alignatt_metadata per update.

**Loop replay on the canonical submission path**
(`outputs/night1_ende_punct_chunk450_instrumented/`):

| Gate                       | F1    | n (updates) |
|----------------------------|-------|-------------|
| `alignatt:rewind`          | **1.000** | 26      |
| `alignatt:source_frontier` | **1.000** | 40      |

That makes **four** artifacts with F1 = 1.000 loop-replay recovery
across two language directions (en→de, cs→en) and two MT backends
(Transformers, vLLM). The paper's Step 7 "observer contract is
complete" claim is now validated on the submission path itself.

### Step 7 v5: multi-feature logistic regression closes the spectrum

Added `scripts/multi_feature_rewind_classifier.py` — L2-regularised
logistic regression over the full 17-feature per-update vector
(provenance averages + positional features + monotonicity features),
evaluated with stratified 5-fold cross-validation so the reported F1
is out-of-fold, not in-sample.

Cross-artifact rewind F1:

| Artifact                              | n_pos | 5-fold CV (default thr) | 5-fold CV (best thr) |
|---------------------------------------|-------|-------------------------|----------------------|
| en→de punct chunk450 (canonical)      |  26   | **0.881**               | **0.926**            |
| cs→en Transformers MT                 |  64   | 0.699                   | 0.706                |
| cs→en vLLM MT (fixed path)            |  38   | 0.606                   | 0.627                |
| en→de K3@700 (n=10, noisy)            |  10   | 0.333                   | 0.419                |

Top weighted features on the canonical en→de artifact:

| Feature                             | Weight |
|-------------------------------------|--------|
| `max_drop_vs_prev_non_none`         | +1.385 |
| `max_backward_jump`                 | +1.385 |
| `unsafe.source_inaccessible`        | −1.165 |
| `source_unit_count`                 | +1.039 |
| `accepted_mean.source_inaccessible` | −1.036 |
| `unsafe.non_source_prompt`          | +0.868 |

Weights match the 2-feature rule discovered in v3
(`max_backward_jump ≥ 9 AND unsafe.source_inaccessible ≤ 0`): large
backward drop + absence of source_frontier-style spillover predicts
rewind.

**Full complexity-vs-fidelity spectrum for the rewind gate:**

| Classifier                              | rewind F1 (realistic n_pos ≥ 26) |
|-----------------------------------------|----------------------------------|
| 1-feature single threshold              | ≤ 0.75                           |
| 2-feature AND / OR combination          | ≤ 0.73                           |
| **Multi-feature (17) logistic, L2 CV**  | **0.63-0.93** (highly artifact-dependent) |
| Loop replay (exact policy semantics)    | **1.000** (all four artifacts)   |

Multi-feature logistic **does** lift over 1-feature on the canonical
submission path (0.93 vs ≤ 0.75), strongly validating that rewind
depends on a *combination* of features rather than any single one.
But it does **not** lift on cs→en at all (0.63-0.70, the same
plateau as 1-feature), showing that on harder artifacts even a
17-feature linear model saturates. Only the loop-replay predictor,
which uses the exact sequential semantics of the policy, hits
F1 = 1.0 reliably across every artifact.

**Paper narrative this closes out:**

> The three-gate MT AlignAtt policy is recoverable from observer
> metadata, but only via replay of its sequential loop. A single
> provenance scalar absorbs `source_frontier` (F1 0.91-0.98 via one
> threshold, F1 0.93-0.98 via multi-feature) but cannot absorb
> `rewind`: single scalars cap at F1 ≤ 0.75, multi-feature logistic
> lifts to F1 0.93 on clean data and only 0.63-0.70 on harder
> artifacts. Loop replay is the only reliable classifier. This
> defines the scope of the "continuous confidence" paper pitch
> cleanly: one gate is scalar-reducible, one is loop-bound, and the
> observer's per-update contract is informationally complete —
> bounded only by the expressive power of the classifier we apply.

Artifacts: `outputs/night1_*/multi_feature_classifier_alignatt_rewind.txt`.
Source: `scripts/multi_feature_rewind_classifier.py`.

### Canonical submission-path per-gate numbers

Ran the v2 per-gate-separability analysis on the freshly-instrumented
canonical artifact (`night1_ende_punct_chunk450_instrumented`) to
close out the paper numbers on the actual submission path:

| Gate                       | Top 1-feature threshold                       | F1      | loop-replay F1 |
|----------------------------|-----------------------------------------------|---------|----------------|
| `alignatt:source_frontier` | `unsafe.source_inaccessible ≥ 0.002`          | **0.988** | 1.000          |
| `alignatt:rewind`          | `max_drop_vs_prev_non_none ≥ 9`               | **0.912** | 1.000          |

**Surprise: on the canonical submission path, single-feature rewind
F1 jumps to 0.91** (vs 0.67-0.75 on the K=3 mechanism-branch
artifacts and cs→en artifacts). The likely reason: `punctuation_lcp`
commits at sentence boundaries, so the updates that reach the MT
loop are more homogeneous — rewind and source_frontier sit on
clearly distinct sides of the same feature axis.

That sharpens the paper conclusion further: on the primary
submission path the scalar approximation is surprisingly close —
F1 0.99 / 0.91 across the two unsafe gates, with loop replay as the
exact reference. The gap between scalar and exact is smaller on
cleaner policy states (pure-punctuation commits) than on mechanism
ablation states (frontier-family commits, which admit more
ambiguous token orderings).

Artifact: `outputs/night1_ende_punct_chunk450_instrumented/per_gate_separability_v2.txt`.

### Multi-clip replication of the canonical-path finding

Ran the same canonical config on a second test-set clip
(`OiqEWDVtWk.wav`, en→de, `punctuation_lcp`, chunk_ms=450,
Transformers MT) to check whether the rewind F1 0.91 / source_frontier
F1 0.99 finding is clip-specific.

| Clip                           | source_frontier F1 | rewind F1 | loop-replay F1 (both) |
|--------------------------------|--------------------|-----------|-----------------------|
| `ccpXHNfaoy.wav` (clip 1)      | **0.988**          | 0.912     | 1.000                 |
| `OiqEWDVtWk.wav` (clip 2, new) | **0.968**          | 0.792     | 1.000                 |

- **`source_frontier` 1-feature scalar is robust across clips**
  (F1 0.96-0.99). The scalar approximation is a submission-grade
  drop-in candidate for this gate.
- **`rewind` 1-feature scalar is clip-dependent** (F1 0.79 vs 0.91);
  both values still above the 0.67-0.75 mechanism-branch cap, but
  with visible variance.
- **Loop replay remains exactly 1.000 on both clips for both gates.**

The paper-level conclusion survives the second-clip check with a
small qualifier: scalar approximation quality for `rewind` varies
clip-to-clip even on a single config, while `source_frontier`
scalar quality is stable. Loop replay is the only invariant-across-
clips method.

Quality numbers on clip 2:

| Metric           | Value    |
|------------------|----------|
| BLEU             | 27.60    |
| chrF             | 63.98    |
| COMET            | 0.832    |
| LongYAAL CU      | 1948 ms  |
| LongYAAL CA      | 2599 ms  |
| RTF              | 0.993    |

Close to clip 1 on BLEU/chrF (within ~0.6). COMET 0.03 lower
(different speaker, topic). Artifact:
`outputs/night1_ende_punct_chunk450_OiqEWDVtWk_instrumented/`.

### Scalar-substitution drift: gate F1 ≠ policy fidelity

The per-gate F1 numbers test whether a scalar can *classify* a gate
correctly *given an update has one*. The paper-grade question is
whether a scalar substituted inside the online policy loop produces
the same commit decisions. Those are not the same question —
inside a loop, substituting a classifier changes WHICH token the
loop `break`s at, which can cascade to different accepted prefixes.

`scripts/scalar_substitution_drift.py` does an offline what-if:
replay two loops per update (exact discrete, and scalar with
`source_frontier := unsafe.source_inaccessible ≥ 0.002`), compare
accepted-token counts and final stop reasons.

Cross-artifact drift:

| Artifact                                       | Updates agree | Aggregate token drift | Direction |
|------------------------------------------------|---------------|-----------------------|-----------|
| en→de punct chunk450 (clip 1, canonical)       | 293/335 = **87.5%** | **−8.3%** (−179 tok) | scalar more conservative |
| en→de punct chunk450 (clip 2 OiqEWDVtWk)       | 203/247 = **82.2%** | **−11.8%** (−172 tok) | scalar more conservative |
| en→de stable_and_accessible K=3 chunk700       | 269/344 = 78.2% | +6.9% (+150 tok) | scalar slightly aggressive |
| cs→en vLLM MT                                  | 155/280 = 55.4% | −24.0% (−327 tok) | scalar very conservative |
| cs→en Transformers MT                          | 180/379 = 47.5% | −41.3% (−781 tok) | scalar very conservative |

**Paper-grade finding:** gate-level F1 0.97-0.99 **does not** mean
scalar substitution is a drop-in replacement. Even on the canonical
submission path, 12-18% of per-update commit decisions change when
the source_frontier discrete gate is replaced by its single-feature
scalar approximation. The substitution typically skews **more
conservative** on en→de (−8% to −12% accepted tokens) and
**much more conservative** on cs→en (−24% to −41%). The frontier
family is more ambiguous: en→de K3@700 drifts slightly more
aggressive (+7%).

**Why:** F1 asks "when the gate fires, does the classifier agree?";
policy fidelity asks "does the loop break at the same token?".
Those differ because:
1. A scalar can fire as source_frontier at an earlier token index
   than the discrete gate would (false positive at a position that
   predates the real firing point).
2. When discrete rewind fires, scalar may still "see"
   source_inaccessible earlier and misclassify — even if the
   gate-level F1 is still high, the cascade through the loop is
   different.

**Paper conclusion refined:** loop replay remains the only
fidelity-preserving offline analysis; per-gate scalar F1 is a useful
upper bound on how close a scalar approximation can get, but
policy-level substitution requires measuring drift in the full
loop context. The ~12-18% canonical-path drift is the concrete
number that quantifies the approximation gap.

Artifacts: `outputs/night1_*/scalar_substitution_drift.txt`.
Source: `scripts/scalar_substitution_drift.py`.

### Threshold sweep: 0.002 isn't optimal, 0.01-0.02 is the sweet spot

The v6 drift numbers used the per-gate-optimal threshold
(0.002, from the F1 0.99 single-feature classifier). That turns
out to be the wrong target: per-gate F1 optimises "classify firings
correctly", drift optimises "match the loop's accepted-token count".
Sweeping thresholds (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
0.1) on the same artifacts:

| Artifact                                        | Agree @ 0.002 | Agree @ best thr | Best thr | Token δ @ best |
|-------------------------------------------------|---------------|------------------|----------|----------------|
| en→de punct chunk450 clip 1 (canonical)         | 87.5%         | **91.3%**        | 0.05     | +2.8%          |
| en→de punct chunk450 clip 2 (OiqEWDVtWk)        | 82.2%         | **83.4%**        | 0.01     | −3.9%          |
| cs→en vLLM MT                                   | 55.4%         | **63.2%**        | 0.02     | +8.2%          |
| en→de K3@700                                    | 78.2%         | **78.5%**        | 0.02     | +6.9%          |

The minimum absolute-token drift sits around threshold **0.01-0.02**
on every artifact — for both canonical clips the aggregate token
delta drops from −8 to −12% (at 0.002) to within ±3% (at 0.02),
while update agreement bumps to 83-91%.

**Refined paper narrative:** the scalar substitution calibrated at
the per-gate-F1-optimal threshold and calibrated at the policy-drift-
optimal threshold are two different calibrations. At the
policy-optimal threshold (0.01-0.02 for `unsafe.source_inaccessible`),
the canonical path's aggregate commit behaviour is within ~3% of
the exact discrete gate; per-update agreement is 83-91%. Still not
bit-identical — loop replay remains the only F1 = 1.0 method — but
close enough that the scalar substitution is a defensible
approximate mechanism for the paper, when the threshold is chosen
by drift minimisation rather than by per-gate F1.

Artifacts: `outputs/night1_*/scalar_threshold_sweep.txt`.
Source: `scripts/scalar_threshold_sweep.py`.

### Step 6 findings: min_source_mass sweep + emit_policy A/B

All at chunk_ms=450 on ccpXHNfaoy.wav with qwen_forced +
gemma_vllm_alignatt + punctuation_lcp.

**min_source_mass sweep (emit_policy=raw_passthrough):**

| min_source_mass | BLEU  | chrF  | COMET | LongYAAL CU | LongYAAL CA | RTF   |
|-----------------|-------|-------|-------|-------------|-------------|-------|
| 0.0 (baseline)  | 27.51 | 63.54 | 0.861 | 1766 ms     | 1466 ms     | 0.393 |
| 0.1             | 28.25 | 63.81 | 0.867 | 2396 ms     | 2140 ms     | 0.466 |
| 0.2             | 28.95 | 63.92 | 0.869 | 2476 ms     | 2197 ms     | 0.443 |

Each +0.1 in min_source_mass buys ~+0.7 BLEU at the cost of ~+700 ms
CA. Latency-quality trade is strictly worse than the `chunk_ms 450→700`
trade (+10.7 BLEU for +1479 ms CA). min_source_mass is a valid knob
on the Pareto front but it is dominated by chunk_ms for simultaneous
submissions — the chunk knob gets you further up the BLEU curve per
millisecond of CA spent.

Reproduces the earlier `phase5_v1_ende_minmass*` sweep qualitatively
under the hardened runtime: BLEU scales similarly (phase5 saw 28.14 /
29.58 at min_mass 0.1 / 0.2 vs our 28.25 / 28.95), with lower absolute
CA on the new path (our 2140 / 2197 ms vs phase5's 2340 / 2788 ms) —
the simulstream path is faster at the same knob setting.

**Emission policy A/B (min_source_mass=0):**

| emit_policy                            | BLEU  | chrF  | COMET | LongYAAL CU | LongYAAL CA |
|----------------------------------------|-------|-------|-------|-------------|-------------|
| `raw_passthrough`   (baseline default) | 27.51 | 63.54 | 0.861 | 1766 ms     | 1466 ms     |
| `freeze_nonexpanding_major_rewrites`   | 27.51 | 63.54 | 0.861 | 1773 ms     | 1484 ms     |

**Bit-identical BLEU / chrF / COMET.** This is the expected outcome: the emit
policy suppresses mid-stream flicker for display purposes but does not
change the committed final translation. CU / CA shift by ~10–20 ms,
which reflects when partial hypotheses are re-emitted (or suppressed)
along the way, not the final content. The A/B confirms that the paper's
quality claims are invariant to the emission policy choice; the policy
only affects display smoothness / LongYAAL-computation timings.

**Practical outcome for the submission:**
- The canonical submission uses `punctuation_lcp` + `raw_passthrough`
  + `min_source_mass = 0.0` + `chunk_ms = 450 or 700`.
- `min_source_mass` is a valid ablation knob for a paper latency-
  quality curve but not a submission default; use `chunk_ms` as the
  primary latency knob as already recommended.
- `freeze_nonexpanding_major_rewrites` stays as an emission-policy
  option for downstream display, not a quality-affecting knob.
