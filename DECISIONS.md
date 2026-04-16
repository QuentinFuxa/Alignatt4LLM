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
