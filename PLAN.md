# PLAN.md

# Gemma-Only Future: Status + Next for vLLM-Native qk_fast AlignAtt

## Critical Review (2026-04-16)

This section supersedes the optimistic reading one might get from the
latest streaming-prefix experiment if read too quickly.

### What is solid and should be kept

- The **non-streaming** `gemma_vllm_qk_fast` seam is now a real system
  result, not a sketch:
  - explicit opt-in backend name
  - compact observer contract preserved
  - custom `worker_cls` bootstrap before warmup / cudagraph capture
  - end-to-end SimulStream runs exist on two clips
  - positive RTF gap vs `gemma_onepass_qk_fast` on both clips
- The **custom worker + tensor observer + deferred warmup** design looks
  like the right foundational systems move. It is narrow, measurable,
  and aligned with the paper-defensible goal of recovering the observer
  engine-natively instead of through late Python hooks.
- The **diagnostic substrate** added around that seam is also good:
  - repeat-on-same-engine harness
  - decode-drift diagnostics
  - prompt-observer cache diagnostics
  - single-audio seam comparison tooling

### What is not yet defensible

- The new **ASR streaming prefix** branch is still only a promising
  experiment, not part of the migration story yet.
- The current `smoke18` gain (`RTF 2.305 -> 1.387`) is **not yet a clean
  ablation** of prefix-prefill alone. The streaming branch changed two
  things at once:
  1. it injects a rolled-back text prefix
  2. it changes the vLLM invocation path from `llm.chat(...)` with a
     temp audio file to `llm.generate(prompt_token_ids=...,
     multi_modal_data=...)`
- The currently checked-in `smoke18` artifacts for non-streaming and
  streaming also come from **different git SHAs**, so the speed gap is
  directionally interesting but not yet the kind of same-code A/B we can
  defend in a paper.
- The streaming branch currently depends on the old **punctuation-based
  commit rule** in `CascadeSession.transcribe_audio`. On longer clips,
  that makes the branch fail structurally: if Gemma continuation does not
  insert sentence-final punctuation, commits never fire, `utt_timestamps`
  does not advance, and the growing audio slice eventually hits Gemma's
  30 s encoder cap.
- The runtime does **not** yet fail fast if
  `asr_streaming_prefix_enabled=True` is used with an unsupported backend.
  Today that mismatch is caught only later through
  `NotImplementedError`.
- The new streaming state machine is **not** protected by targeted tests
  yet. The existing test suite passing is good, but it mostly validates
  the observer seam, not the new ASR prefix-carry logic.
- The design docs outside this file are now partially stale: they still
  describe the public runtime surface as two frontends only. Treat this
  file and the actual code as the ground truth until the docs are synced.

### Revised next steps

1. **Run a clean same-SHA, same-audio control for the streaming claim.**
   Before any further design conclusions, compare on `tmp/alignatt_smoke18.wav`:
   - non-streaming `gemma_vllm_qk_fast`
   - a control path using the same `llm.generate(prompt_token_ids + multi_modal_data)`
     invocation but with **no** injected text prefix
   - the actual prefix-prefill path
   The goal is to separate:
   - benefit from the API / input-path change
   - benefit from prefix-prefill itself
   Do not quote `1.387` as "the" prefix-prefill gain until this control exists.

2. **Add fast-fail validation and focused tests around the streaming branch.**
   Minimum scope:
   - reject `asr_streaming_prefix_enabled=True` unless
     `alignment_backend_name == "gemma_vllm_qk_fast"`
   - test `_compute_streaming_prefix()` on repeated words and trailing punctuation
   - test streaming-state reset on utterance commit / session reset
   - test that the returned streaming `AlignmentResult` still satisfies the
     word-count invariant the commit logic relies on
   These are lightweight and high-signal; they protect the new stateful logic
   without bloating the repo.

3. **Keep the main migration story centered on the non-streaming vLLM seam.**
   Until step 1 is clean and the commit signal problem is solved, the
   primary claim remains:
   - `gemma_vllm_qk_fast` non-streaming is a valid experimental backend
   - it is faster than `gemma_onepass_qk_fast`
   - it keeps the compact AlignAtt observer contract
   Do not let the still-fragile streaming branch redefine the repo's main
   success criterion prematurely.

4. **Only then decide whether the right next research move is stability-based commit.**
   The "stability-based commit" idea is still the most principled current
   candidate for making prefix-prefill usable beyond punctuation-friendly
   models. But it should be opened only after the same-code control above
   confirms that the remaining speed win is really due to prefix-prefill.
   Otherwise we risk solving the wrong problem.

5. **Sync the docs only after the story above stabilizes.**
   Once the backend/streaming picture is clean, update:
   - `ALIGNATT_LLM.md`
   - `E4B_ALIGNATT_CASCADE_DESIGN.md`
   so they stop contradicting the actual runtime surface.


## Streaming Prefix-Prefill Experiment (2026-04-16) — Status, Gains, Difficulties

This section is the most recent experimental record. Read it after the
critical review above; the older handoff below is still valid but
pre-dates these findings.

### What was built

An opt-in "prompt-prefix streaming" path for `gemma_vllm_qk_fast`,
reusing the idea from Qwen3-ASR's `streaming_transcribe`: carry the
previously decoded text across chunks and inject it as the assistant
turn's prefill so the model only decodes the text delta, not the whole
sentence from scratch. Knobs:

- `CascadeRuntimeConfig.asr_streaming_prefix_enabled` (off by default)
- `CascadeRuntimeConfig.asr_streaming_rollback_words` (default 2)
- `CascadeRuntimeConfig.asr_streaming_unfixed_chunks` (default 2)

Corresponding CLI flag on `run_simulstream_batch.py`:
`--asr-streaming-prefix-enabled`, `--asr-streaming-rollback-words`,
`--asr-streaming-unfixed-chunks`. The code path is fully gated — with
the flag off, behaviour is byte-for-byte identical to the previous
non-streaming `gemma_vllm_qk_fast`.

Word-level rollback is used on purpose (not Qwen-style token-level):
it keeps the prefix ending on a clean word boundary, which preserves
the invariant `find_end_time` relies on (`len(words) ==
len(remove_punctuation(text).split())`). Token-level rollback was
tried first and broke this invariant when the boundary landed
mid-word.

### Measured gain — smoke18 (18 s clip)

On `tmp/alignatt_smoke18.wav` the RTF drops cleanly:

- `gemma_vllm_qk_fast` non-streaming: RTF `2.305`
- `gemma_vllm_qk_fast` streaming prefix: RTF `1.387`
- relative improvement: ~39.8% RTF reduction
- qwen_forced baseline (reference only): RTF `0.798`

Commit cadence also improved: 12 distinct commit frontiers vs. 4
non-streaming, and the emitted German translation reaches
substantially further through the audio in the same wallclock
window. No observer failures. Artifacts:
`outputs/simulstream_gemma_vllm_streaming_v2_smoke18/`.

### Difficulty — the approach does not yet generalise

On `test-set/audio/ccpXHNfaoy_short60s.wav` the streaming run hits a
hard wall: the backend crashes with
`GemmaAudioTooLongError: Audio is 30.150s but Gemma encoder cap is
30.000s`. Gemma's audio encoder is hard-capped at 30 s, and the
cascade only truncates the audio slice when a sentence-level commit
advances `utt_timestamps`. In non-streaming the first commit on this
clip lands at ~22 s of audio (well within the cap); in streaming mode
no commit fires at all before 30 s is reached, so the growing audio
slice overflows the encoder.

Debug tracing (`CASCADE_ASR_STREAMING_DEBUG=1`) shows why: the
streaming ASR hypotheses on this clip never contain a sentence-
terminal period. Samples:

- chunk 5 (4.05 s): `"Hi I'm Si Yuan from Fudan University"` — no period
- chunk 20 (10.80 s): `"...Distinctness script knowledge from large language models"` — no period
- chunk 44 (21.60 s, where non-streaming commits): `"...step-by-step instructions in the form of generated scripts"` — no period
- through chunk 62 (29.70 s): still no period anywhere in the hypothesis

Because the LCP-based commit logic only fires on
`n_utterances(asr_segment) >= 1` (which requires `.`, `!`, or `?` in
the LCP), commits never trigger and the audio slice grows unbounded
until the 30 s cap blows up.

### Why the approach does not produce periods on Gemma

The prompt-prefix continuation is structurally unable to insert
retroactive punctuation. At chunk N, the prefix is the previous
hypothesis minus a short rollback tail; the continuation only covers
that rollback tail plus any new words at the current audio frontier.
A sentence boundary that *should* have been placed several chunks ago
(e.g. after "University" at ~5 s) is already frozen deep inside the
prefix and the model has no mechanism to rewrite it. Non-streaming
avoids this because it regenerates the entire transcription each
chunk, so once the audio is long enough the model can re-decide
globally where periods go.

### Hypotheses tested and ruled out

- **"Gemma has a streaming-only mode"** — wrong. Both Qwen3-ASR and
  Gemma are stateless multimodal LLMs; "streaming" is just a harness
  around repeated inference calls.
- **"It's a chat-template quirk with `continue_final_message=True`"**
  — also wrong. We replicated Qwen3-ASR's exact call shape:
  `llm.generate(prompt_token_ids=user_turn + tokenize(prefix_text),
  multi_modal_data={"audio": [audio_np]}, ...)`, bypassing
  `llm.chat()` entirely. Behaviour is byte-identical to the
  `continue_final_message=True` path: same periodless hypotheses,
  same cap crash. So the invocation path is not the variable.
- **"A stronger prompt instruction will bring periods back"** — also
  ruled out. Adding an explicit "preserve sentence punctuation" rule
  to the ASR instruction *did* cause the model to emit periods, but
  it achieved this by hallucinating fake sentences to punctuate
  ("This thing's script knowledge is amazing!" in place of the real
  audio). That is an AGENTS.md "screugneugneu" violation and is not
  defensible — reverted.

### What this tells us about Qwen3-ASR vs Gemma-4

The remaining honest explanation is model behaviour under
prompt-prefix continuation. Qwen3-ASR (dedicated ASR head) inserts
sentence-terminal periods inside its continuation when the audio
justifies it; Gemma-4 (general multimodal LLM) does not reliably do
so on this input. That is a property of the model weights, not of
the inference plumbing.

### Options for a defensible path forward

1. **Stability-based commit** — principled, model-agnostic. Replace
   the "LCP must contain `.`, `!`, or `?`" rule with "LCP has been
   stable for N consecutive chunks past some minimum length → commit
   at that boundary". This decouples commit semantics from the
   model's punctuation behaviour and keeps the prefix-prefill
   decode-delta win. The claim we could defend in a paper is:
   *stability-based acceptance is a more fundamental streaming-ASR
   signal than punctuation, and it is what actually makes
   prompt-prefix streaming usable across models that were not
   trained for it.*

2. **Abandon prefix-prefill; use prior text as hint context inside
   the user turn** (e.g. `"Transcript so far: {prev_text}. Continue
   transcribing from the audio."`). Model regenerates fresh each
   chunk, so punctuation is produced correctly, but decode is back
   to O(full sentence) — we lose the smoke18 RTF gain and end up
   roughly where non-streaming was.

3. **Do nothing under this track and move to the KV-cache-native
   prompt observer (step 1)**. This merges back into the other open
   work item and does not directly address the repeated-decode
   bottleneck.

Current recommendation: option 1. It is a small, localised change to
the cascade's commit logic, compatible with both streaming and
non-streaming backends, and it is the finding that would carry a
research paper — "how to make prompt-prefix streaming work for ASR
models that do not emit punctuation in continuation mode".

### What is in the repo right now

- Streaming prefix-prefill is implemented, gated, tested to import
  and to pass the existing 32 unit tests.
- smoke18 end-to-end run under streaming succeeded with RTF `1.387`
  and coherent German translation.
- Longer clips (short60s) crash under streaming because of the
  commit-signal incompatibility described above. This is not an
  observer bug, not a vLLM bug, and not a chat-template bug.
- The non-streaming `gemma_vllm_qk_fast` path is untouched and still
  the validated reference (RTF `2.305` on smoke18, `2.580` on
  short60s).


## Handoff (2026-04-16) — For the Next Agent

**Read this section first.** Everything below is historical context.

### Current state (what is actually true right now)

- three alignment backends are wired in the runtime:
  `qwen_forced`, `gemma_onepass_qk_fast` (both stable),
  `gemma_vllm_qk_fast` (experimental, opt-in)
- the canonical SimulStream comparison still defaults to the two
  stable backends only
- 32 unit tests in `test_alignment_helpers.py` pass
- `gemma_vllm_qk_fast` has two validated end-to-end SimulStream runs:
  - `tmp/alignatt_smoke18.wav` (18 s):
    RTF `2.305` vs `gemma_onepass_qk_fast` `2.950` — about 21.9% faster.
    Artifact: `outputs/simulstream_gemma_vllm_integration_smoke18/`.
  - `test-set/audio/ccpXHNfaoy_short60s.wav` (60 s, 3.3× longer,
    different content):
    RTF `2.580` vs `gemma_onepass_qk_fast` `3.100` — about 16.8% faster.
    Artifacts: `outputs/simulstream_gemma_vllm_short60s/` and
    `outputs/simulstream_gemma_onepass_short60s/`.
  - still much slower than `qwen_forced` RTF `0.798` on `smoke18`.
- validated vLLM seam defaults: `worker_cls=gemma_vllm_worker.GemmaAlignAttWorker`,
  `cudagraph_mode="full"`, `enforce_eager=False`,
  `enable_prefix_caching=False`

### What is done

- **Step 3** (gated runtime integration seam) — DONE, validated end-to-end
- **"What To Do Now" steps 1 and 3** — DONE (bootstrap hardening,
  one-clip comparison automation)
- initial evidence for **Step 4** (end-to-end gains) on one clip

### What still lacks

In rough priority order for a paper-defensible result:

1. **Multi-audio generalization.** The two-clip sanity check is now
   passed (`smoke18` at 18 s and `ccpXHNfaoy_short60s` at 60 s, both
   show a positive RTF gap for `gemma_vllm_qk_fast` over
   `gemma_onepass_qk_fast` with no observer failures). The next
   widening, a 5–10 clip sanity set, is not yet done. The direction
   is confirmed, the magnitude (16.8%–21.9%) is not yet well
   characterised across clips.

2. **Step 1: prefix-caching semantic gap.** Still open. Identical hot
   replays drift in text when prefix caching is on, even though
   observer completeness is restored by the host-side cache. No
   KV-cache-native prompt observer path exists. The current default
   (prefix caching off) works but leaves a real performance ceiling
   on the table. The `_compute_decode_drift` instrumentation is in
   place for this investigation.

3. **Step 5: incremental ASR track.** The remaining RTF gap from
   `gemma_vllm_qk_fast` (2.305) to `qwen_forced` (0.798) is now larger
   than the backend-decode gap this work closed. That strongly
   suggests the SimulStream-level repeated prefix rerun is now the
   dominant cost, not observer reconstruction. This should be opened
   as its own project.

4. **Paper-level framing.** The current defensible claim is:
   *"A multimodal causal LLM's ASR-side AlignAtt observer can be
   recovered engine-natively under `cudagraph=full` via a custom
   worker and tensor-buffer observer, with a compact per-token
   contract and no Python-side side effects during compiled
   execution."* This is a systems result. For a paper, it needs to
   either (a) scale to multi-audio evaluation with a proper metric
   story, or (b) be the engine underneath a sharper research claim
   (e.g. what compact observer statistics are actually sufficient
   for monotone streaming acceptance).

### Suggested next concrete action

The second-clip sanity check is done (see the two-run table above).
The plan's own evidence says the next lever with the larger remaining
end-to-end RTF headroom is **Step 5: the incremental-ASR track**, not
more backend-observer work. That track is now being opened as a
separate investigation, with the rule from the "What not to do" list
still honoured: observer work and incremental-ASR work stay as two
independent questions with two independent success criteria.

A full 5–10 clip sanity set is still pending but should not hold up
the incremental-ASR investigation. Treat it as orthogonal multi-audio
generalization work to run alongside, not a blocker.

### What not to do

- do not silently make `gemma_vllm_qk_fast` the default
- do not re-enable prefix caching without a KV-cache-native observer
- do not widen to a full benchmark sweep before the two-clip sanity
  check is clean
- do not conflate observer work with incremental-ASR work


## Why this document now exists

The repo's canonical SimulStream runtime still has exactly two active
frontends:

- `qwen_forced`
- `gemma_onepass_qk_fast`

The research question is no longer "should we try vLLM for Gemma?".

It is:

**Which vLLM seam actually survives a real engine-native execution path,
while preserving a compact, paper-defensible AlignAtt observer on the
ASR side?**

That question must stay separate from the other system issue that still
exists in the runtime:

- ASR-side AlignAtt observation
- rerun of the uncommitted audio prefix inside SimulStream

This document is now a living status note for that effort, not a
speculative implementation plan.


## Current baseline to keep in mind

Canonical runtime comparison artifacts already exist:

- `outputs/simulstream_compare_smoke18/comparison_report.json`
- `outputs/simulstream_compare_smoke18/qwen_forced/`
- `outputs/simulstream_compare_smoke18/gemma_onepass_qk_fast/`

On `tmp/alignatt_smoke18.wav`, the current runtime reading remains:

- `qwen_forced`
  - WER `0.1143`
  - CER `0.0308`
  - first non-empty emission `4.05 s`
  - wallclock `14.37 s`
  - RTF `0.798`
- `gemma_onepass_qk_fast`
  - WER `0.1714`
  - CER `0.2000`
  - first non-empty emission `2.70 s`
  - wallclock `53.09 s`
  - RTF `2.950`

The important reading is unchanged:

- Gemma does not have a load-time problem here.
- Gemma has a runtime frontend cost problem.
- That cost problem is not explained by `qk_fast` math alone.


## Validated

These points are now established by actual repo work and single-clip GPU
validation on `tmp/alignatt_smoke18.wav`.

### 1. The compact observer contract is real

The backend-to-runtime observer surface has been reduced to compact,
structured per-token statistics. The runtime does not need giant
attention dumps to make monotone acceptance decisions.

That means the defensible target is:

- token identity
- aligned source/audio position
- optional accessible mass / compact provenance metadata
- optional blocked-region metadata

Not:

- full attention tensors in Python
- full per-head dumps as a public runtime dependency

### 2. The current Gemma Transformers path is dominated by repeated decode

Fine-grained timing added to the current Gemma path showed that the main
cost is repeated Gemma decode, not `qk_fast` reconstruction itself.

On the diagnostic single-audio run, the important reading was:

- total backend around `3810 ms`
- prompt forward around `630 ms`
- decode step total around `2821 ms`
- `qk_fast` reconstruction around `244 ms`

So the large system problem is repeated decode on a growing prefix, not
some hidden catastrophe inside the reconstruction math.

### 3. An experimental vLLM Gemma ASR observer backend now exists

There is now an experimental backend that can produce:

- transcript text
- word timings
- compact observer payload

without depending on the current Transformers probing loop as its only
execution path.

Relevant experimental artifacts include:

- `tmp/alignatt_smoke18_gemma_vllm_eager_baseline.json`
- `tmp/alignatt_smoke18_gemma_vllm_custom_worker_full_cg.json`
- `tmp/alignatt_smoke18_gemma_vllm_compile_no_cg_buffer_module.json`
- `tmp/alignatt_smoke18_gemma_vllm_custom_worker_full_cg_buffer_module.json`
- `tmp/alignatt_smoke18_gemma_vllm_compile_repeat_same_engine.json`
- `tmp/alignatt_smoke18_gemma_vllm_full_cg_repeat_same_engine.json`
- `tmp/alignatt_smoke18_gemma_vllm_compile_repeat_same_engine_prefix_cache.json`
- `tmp/alignatt_smoke18_gemma_vllm_full_cg_repeat_same_engine_prefix_cache.json`

### 4. `worker_cls` + on-device tensor observer survives `cudagraph=full`

The key positive result is no longer hypothetical:

- a custom `worker_cls` was able to install the observer before the
  observer-aware warmup/capture
- the observer was rewritten as tensor-buffer state on device
- that path now survives a real `cudagraph=full` run

This is the first genuinely positive engine-native seam in this repo for
Gemma ASR-side AlignAtt.

### 5. The full-cudagraph run recovered the expected observer signal

On `tmp/alignatt_smoke18_gemma_vllm_custom_worker_full_cg_buffer_module.json`,
the observer payload is not empty or partial. The important numbers are:

- `effective_head_count=8`
- `generated_token_count=45`
- per-layer `decode_q_count=45`
- per-layer `prompt_audio_capture_count=450`

That means the backend is not merely "running under vLLM". It is
actually recovering the Q/K material required for `qk_fast`.

### 6. Full cudagraph is already better than the eager vLLM baseline

The positive result is also performance-relevant, not just functional.

On the same clip:

- eager vLLM baseline backend: about `4202 ms`
- custom worker + `cudagraph=full`: about `3267 ms`

The important interpretation is:

- this seam is already faster than the eager observer baseline
- the gain is not coming from transcript heuristics or semantic shortcuts
- the design direction changed because of a real measurement, not taste

### 7. Compile-only can also recover the observer if the observer exists
before engine build

The earlier compile-only failure was not just "compiled execution hates
observer buffers" in the abstract.

The more precise reading is:

- late observer installation after engine initialization is not enough
- compile-only can recover the compact observer if the per-layer tensor
  observer module is already attached when the worker loads the model
- reconfiguration must reuse that same module instead of replacing it

On `tmp/alignatt_smoke18_gemma_vllm_compile_no_cg_buffer_module.json`,
the important numbers are:

- `effective_head_count=8`
- `generated_token_count=46`
- `forward_call_count=147`
- per-layer `decode_q_count=46`
- per-layer `prompt_audio_capture_count=450`
- total backend about `3576 ms`

So compile-only is no longer a pure negative control on this clip. It is
now a second positive engine-native seam, though still slower than full
cudagraph.

### 8. Same-engine repeated requests are stable if prefix caching stays off

The next important correction is about repeated requests on the same
engine, not just a single cold call.

With the current observer implementation:

- bootstrap-before-engine-build is necessary but not sufficient
- prompt-side Q/K capture still depends on a real prompt forward
- prefix caching can bypass that prompt forward and silently remove the
  prompt-side observer signal

On the repeat artifacts:

- `tmp/alignatt_smoke18_gemma_vllm_compile_repeat_same_engine.json`
- `tmp/alignatt_smoke18_gemma_vllm_full_cg_repeat_same_engine.json`

the stable reading is:

- with `enable_prefix_caching=False`, both `vllm_compile` and
  `cudagraph=full` keep `effective_head_count=8`
- both keep per-layer `prompt_audio_capture_count=450`
- both keep per-layer `decode_q_count=45`
- repeated hot runs stay text-stable on the same clip

The hot backend timings on that same clip are now roughly:

- compile-only hot runs: `1127-1156 ms`
- full-cudagraph hot runs: `592-593 ms`

So the observer seam is now robust across repeated same-engine requests,
but only under a prompt-forward-compatible serving mode.

### 8.5. An explicit prompt-observer cache can restore observer completeness
under prefix caching, but not text stability

The repeat story changed again once the backend grew a host-side cache
for prompt-side observer keys.

On the prefix-cache repeat artifacts:

- `tmp/alignatt_smoke18_gemma_vllm_compile_repeat_same_engine_prefix_cache.json`
- `tmp/alignatt_smoke18_gemma_vllm_full_cg_repeat_same_engine_prefix_cache.json`

the important reading is:

- hot same-engine runs no longer execute a real prompt forward
- `prompt_forward_call_count` drops from `3` to `0`
- the prompt-observer cache hits and restores all `3` captured layers
- per-layer `prompt_audio_capture_count` still returns to `450`
- both seams keep `effective_head_count=8` after restore

So prefix caching does not have to imply observer incompleteness if the
backend carries an explicit prompt-observer cache keyed to the repeated
request state.

But the other half of the result is just as important:

- compile-only hot runs change from `44` observer tokens to `45`
- full-cudagraph hot runs change from `44` observer tokens to `53`
- the hot repeated text surface is not identical to the cold run
- on the `full-cg` repeat artifact, the vLLM-side prompt token ids still
  match the locally built prompt on every run, so this no longer looks
  like a simple prompt-assembly mismatch

So this host-side prompt-observer cache is evidence that observer
completeness can be recovered under prefix caching on repeated identical
prompts. It is not yet evidence that prefix-cached serving is
semantically stable enough to become the default research path.


## False Paths / Invalidated Hypotheses

These paths should now be considered invalidated, or at least false as
the primary direction.

### 1. Late Python hooks are not an engine-native answer

Patching `Gemma4Attention.forward` from Python and hoping that the hook
will survive real compiled or replayed execution is a false path.

In particular:

- `preload_class` / `postload_instance` hooks are not sufficient as the
  final engine-native mechanism
- they may work in eager mode
- they do not define a robust path through real vLLM execution modes

### 2. `torch.compile` and CUDA graph do not reward Python-side side effects

The old Python-hook observer established two different failures:

- compile-only could bypass the observer entirely
- CUDA graph replay could stop re-entering the Python hook for decode

So "the hook worked in eager, therefore it is close to engine-native" is
no longer a tenable interpretation.

### 2.5. Prefix caching is not free for the current observer path

The current experimental backend still reconstructs prompt-side observer
keys from the actual prompt forward, not from a KV-cache-native key
reader.

So the following statement is currently false:

"Once the observer is bootstraped early, prefix caching is harmless."

What the repeat runs showed instead is:

- if prefix caching is enabled, the prompt forward can disappear
- decode-side Q capture may still exist
- without an explicit prompt-observer cache, prompt-side K capture can
  drop to zero and the compact observer becomes incomplete
- with an explicit prompt-observer cache, observer completeness can be
  restored on repeated identical prompts, but the decoded text surface
  can still change between cold and hot runs

### 3. Dynamic indexing during capture is unsafe

The first on-device observer attempt still failed under CUDA graph
capture because dynamic operations such as `nonzero(...)` broke capture.

That is now a known anti-pattern for this line of work.

The observer path that survived capture used fixed-shape tensor writes
and persistent scratch buffers instead.

### 4. "Move Gemma to vLLM" is too vague to be useful

The goal is not "put Gemma under another inference library".

The goal is:

**an ASR-side AlignAtt observer that remains valid under real vLLM
execution modes**

Anything less precise than that leads back to dead-end experiments.

### 5. A faster observer does not, by itself, solve SimulStream latency

Even after a backend win, SimulStream may still rerun too much audio
because the runtime keeps recomputing the uncommitted prefix.

So the following statement should now be treated as false:

"If we make the observer much faster, the runtime problem is basically
solved."

That is not established. The prefix-rerun issue remains a separate
system-level bottleneck.


## Current Best Reading

The best current reading of the system is now:

- `cudagraph=full` with a custom `worker_cls` and on-device tensor
  observer remains the best validated seam on one clip
- `vllm_compile` with `cudagraph=none` also now recovers the compact
  observer on one clip
- the decisive implementation detail is observer bootstrap timing, not
  only the choice of tensor buffers versus Python dicts
- repeated same-engine robustness is now also validated on one clip when
  prefix caching is disabled
- an explicit host-side prompt-observer cache can also recover observer
  completeness under prefix caching on repeated identical prompts for
  both positive seams
- that prefix-cache path is still not text-stable on this clip, so it is
  diagnostic evidence, not yet the clean serving contract
- on `full-cg`, that remaining prefix-cache instability now appears more
  likely to be a decode-path issue than a prompt-construction issue
- the next hard systems question is no longer "can compile-only work?"
  but "how should prompt-side observer state interact with prefix/KV
  caching?"
- the next hard problem is no longer `qk_fast` reconstruction speed
- the next hard problem is making this bootstrap-based observer path
  robust enough to trust across more request shapes without widening the
  benchmark prematurely

So the priority changed.

The current frontier is not:

- optimize `qk_fast` math further
- wire the experimental backend into all runtime surfaces immediately

It is:

- keep the observer attached early enough that compiled execution
  actually sees it
- keep prefix caching disabled by default for this backend until
  prompt-side keys can be read from a cache-native path or another
  semantically stable accepted-state cache
- then tighten the one-clip comparison and robustness story before any
  runtime integration


## What To Do Now

The next sequence should be explicit and narrow.

### 1. Bootstrap hardening + decode-drift investigation tooling — DONE

The worker bootstrap substrate is now hardened:

- per-layer observer module is attached before engine build / warmup /
  compile (existing `GemmaAlignAttWorker.load_model()`)
- later request preparation reuses the same module identity (existing
  `_configure_audio_qk_tensor_observer_on_model` reuse path, tested)
- prefix caching remains off by default
- compact observer contract is unchanged
- **new:** post-warmup observer integrity verification
  (`_verify_observer_integrity` in worker) — after `compile_or_warm_up_model()`
  completes, the worker now verifies all tensor observer bindings survived
  and reports `observer_intact_after_warmup` in install diagnostics; fails
  loudly if compile/cudagraph replaced the attention modules
- **new:** `reset_caches()` on `GemmaVLLMAttentionAlignmentBackend` —
  fulfills the `AlignmentBackend` contract, clears prompt observer cache
  and decode-drift state

The decode-drift investigation tooling is also now in place:

- **new:** `_compute_decode_drift` in the vLLM backend — tracks previous
  run's token IDs and on subsequent requests reports `identical`,
  `prev_token_count`, `current_token_count`, `first_divergence_index`
  in the `decode_drift` diagnostics field
- this is the concrete instrument for investigating why prefix-cached
  hot runs change the decoded surface even when observer completeness
  is restored
- the existing artifacts already show: without prefix caching hot runs
  are byte-identical; with prefix caching the decoded surface drifts,
  worse under full cudagraph; observer completeness is fine in all cases
- the next investigation step is to use this instrumentation to pinpoint
  where in the decode path the divergence originates

### 2. Keep `cudagraph=full` as the validated engine-native reference

Do not treat the new full-cudagraph path as speculative anymore.

It is the current reference experimental backend for:

- engine-native Q/K access
- compact observer recovery
- single-clip parity/performance comparison

### 3. One-clip comparison is now automated — DONE

The minimum comparison set is now runnable from a single command:

- **new:** `seam_comparison` CLI subcommand in `run_alignment_single_audio.py`
  runs all three seams on one audio (eager baseline, `cudagraph=full`,
  `vllm_compile + cudagraph=none`), writes per-seam bundles and a
  `seam_comparison.json` summary with text-agreement check
- **new:** `--repeat N` flag on `gemma_vllm_inspect` runs the same
  request N times on the same engine, writes per-run bundles and a
  `.stability.json` summary reporting `text_stable`, `unique_texts`,
  `token_counts`, per-run `decode_drifts`, and timing variance

Do not widen the benchmark set before the behavior is stable and
well-understood on this single clip.

### 3.5. Only re-open prefix caching through a KV-cache-native observer

If we want prefix caching back for this backend, the next implementation
should not be "hope that prompt capture survives anyway".

It should be:

- recover prompt-side observer keys from the runtime KV cache itself
- or build an explicit prompt-key observer cache that is semantically
  tied to the accepted request state and does not change the decoded
  surface across identical hot replays

Until then, disabling prefix caching remains the cleaner default
experimental choice, even though the host-side cache is now a useful
diagnostic tool.

### 4. Only then decide on runtime integration

The experimental backend should not be wired into the canonical
SimulStream compare path until:

- compile/cudagraph behavior is understood well enough
- the observer remains semantically defensible
- single-clip results are stable and interpretable

### 5. If runtime RTF remains too high, open the separate incremental-ASR task

If backend gains are real but SimulStream still has poor end-to-end RTF,
the next task is not "more vLLM work by default".

It is:

**stateful incremental ASR across chunks**

That should be opened explicitly as a separate system problem, not folded
silently into backend-observer work.


## Operational Status Snapshot (2026-04-16)

This section is the shortest operational answer to:

- where we are now
- whether `vLLM + AlignAtt` already works
- whether it is already the runtime default
- whether it is already faster in a meaningful sense

### Already validated

- a real `vLLM + AlignAtt` Gemma ASR backend now exists in
  `gemma_vllm_alignment_backend.py`
- the backend is not a Python-hook-only artifact anymore; the positive
  seam is `worker_cls` + on-device tensor observer
- `cudagraph=full` is a real positive result, not a speculative path
- `vllm_compile + cudagraph=none` is also a positive result on the same
  mono-audio clip
- the compact observer contract is preserved: the runtime can consume
  compact per-token alignment/provenance statistics without depending on
  giant attention dumps
- repeated same-engine runs are text-stable on `tmp/alignatt_smoke18.wav`
  when `enable_prefix_caching=False`
- on the mono-audio backend benchmark, `cudagraph=full` is already faster
  than the eager vLLM observer baseline and faster than the previous
  Transformers-side Gemma observer timing reported in this document

### Still experimental

- the backend remains mono-audio-first and diagnostic-first
- prompt-side observer recovery under prefix caching still relies on a
  host-side prompt-observer cache, not a KV-cache-native observer path
- with prefix caching enabled, observer completeness can be restored, but
  decoded text is not yet stable across identical hot replays
- the current evidence is still intentionally concentrated on one clip:
  `tmp/alignatt_smoke18.wav`

### Now wired and validated on one clip

- `cascade_runtime.py` now exposes `gemma_vllm_qk_fast` as a third
  valid alignment backend name, alongside `qwen_forced` and
  `gemma_onepass_qk_fast`
- `build_alignment_backend()` builds a
  `GemmaVLLMAttentionAlignmentBackend` when this name is selected
- the default comparison set (`STABLE_ALIGNMENT_BACKEND_NAMES`) still
  only includes the two stable frontends
- `run_simulstream_batch.py` accepts `gemma_vllm_qk_fast` via CLI
- the active `gemma_onepass_qk_fast` runtime path is still the current
  Transformers-based implementation — the vLLM backend does not replace it
- the vLLM backend now has a validated full-SimulStream run on
  `tmp/alignatt_smoke18.wav`: RTF `2.305`, no observer failures

### Not yet true

- the vLLM backend has **only** been validated on one canonical mono-audio
  clip through the full SimulStream loop
- we do **not** yet have a claim that the RTF improvement generalizes
  beyond that one clip
- the default comparison does not include `gemma_vllm_qk_fast`
- prefix-caching semantic gap is still open (step 1)


## Next Execution Order (2026-04-16)

The next steps should stay narrow and ordered. Do not widen the problem
until the current gate is clean.

### 1. Close the prefix-caching semantic gap

Goal:

- explain exactly why identical hot replays drift in text when prefix
  caching is enabled, even though observer completeness is restored

Required outcome:

- either a KV-cache-native prompt observer path
- or a principled proof that prefix-cached serving is semantically stable
  for this backend on repeated identical requests

Stop condition:

- repeated same-engine runs with `enable_prefix_caching=True` are both
  observer-complete and text-stable on the canonical mono-audio clip

### 2. Re-run the one-clip seam comparison only after step 1 changes

Goal:

- keep a strict single-clip benchmark loop while the serving contract is
  still moving

Required outcome:

- fresh `seam_comparison` artifacts for eager baseline,
  `cudagraph=full`, and `vllm_compile + cudagraph=none`
- explicit reading of cold vs hot timing and text agreement

Stop condition:

- the fastest positive seam is still clearly identified after the latest
  backend changes

### 3. Add a gated runtime integration seam, not a silent replacement — DONE

Goal:

- test runtime integration without pretending the migration is complete

Required outcome:

- add an explicit experimental runtime backend name for the Gemma vLLM
  ASR AlignAtt path instead of silently changing `gemma_onepass_qk_fast`
- keep the old path available until the new seam is validated under the
  real SimulStream loop

Implementation:

- `VALID_ALIGNMENT_BACKEND_NAMES` now includes `"gemma_vllm_qk_fast"`
- `STABLE_ALIGNMENT_BACKEND_NAMES` keeps only the two stable frontends
- `build_alignment_backend()` in `cascade_runtime.py` builds a
  `GemmaVLLMAttentionAlignmentBackend` when `alignment_backend_name`
  is `"gemma_vllm_qk_fast"`
- `CascadeRuntimeConfig` carries vLLM-specific defaults that match the
  validated `cudagraph=full` seam: `enforce_eager=False`,
  `enable_prefix_caching=False`, `cudagraph_mode="full"`
- `run_simulstream_compare.py` uses `STABLE_ALIGNMENT_BACKEND_NAMES`
  by default — the experimental backend is not included in the default
  two-backend comparison
- `run_simulstream_batch.py` accepts `gemma_vllm_qk_fast` as a
  `--alignment-backend-name` choice
- the old `gemma_onepass_qk_fast` path is untouched

Integration test (2026-04-16):

- single-clip diagnostic harness run of the integrated backend on
  `tmp/alignatt_smoke18.wav` produces the expected observer signal
- artifact: `tmp/alignatt_smoke18_gemma_vllm_runtime_integration_test.json`
- `effective_head_count=8`, `missing_heads=[]`,
  `generated_token_count=44`, `monotonicity=0.88`, `finish_reason="stop"`
- cold total backend around `3791 ms`, consistent with the validated
  `cudagraph=full` reference (~3267 ms hot baseline)

Full SimulStream validation (2026-04-16):

- end-to-end SimulStream run via `run_simulstream_batch.py` with
  `--alignment-backend-name gemma_vllm_qk_fast` on
  `tmp/alignatt_smoke18.wav`
- artifact directory: `outputs/simulstream_gemma_vllm_integration_smoke18/`
- RTF `2.305`, wallclock `41.48 s`, 20 updates emitted, no observer
  failures, no crashes
- produces a reasonable partial German translation:
  `"Hallo, ich bin Si Yuan von der Fudan"`

Stop condition: **MET** on the canonical mono-audio clip.

- single-clip backend validation: PASS
- full SimulStream loop validation: PASS

### 4. Only then measure end-to-end SimulStream gains — INITIAL RESULT

Goal:

- separate backend wins from system wins

Required outcome:

- compare the experimental runtime seam against the current canonical
  `gemma_onepass_qk_fast` path on `tmp/alignatt_smoke18.wav`
- record whether wallclock / RTF gains survive the full runtime, not
  only the backend harness

Initial result (2026-04-16) on `tmp/alignatt_smoke18.wav`:

- `qwen_forced`: RTF `0.798` (baseline reference)
- `gemma_onepass_qk_fast`: RTF `2.950` (Transformers-based Gemma path)
- `gemma_vllm_qk_fast`: RTF `2.305` (new vLLM-based Gemma path)

The vLLM backend provides a ~21.9% RTF reduction over the current
Transformers-based Gemma path under the full SimulStream loop. It is
still substantially slower than `qwen_forced`, which suggests the
remaining bottleneck is not just backend decode speed but also the
SimulStream-level repeated prefix rerun. That is the separate
incremental-ASR track in step 5.

Stop condition:

- we know whether the main remaining bottleneck is still backend decode,
  runtime prefix rerun, or both
- initial mono-audio evidence suggests: **both**, but the prefix-rerun
  effect is now the larger remaining gap since the vLLM backend already
  closed most of the Gemma-vs-Qwen-decode gap that a full Transformers
  path left open

### 5. Open the incremental-ASR track, reusing Qwen3-ASR's streaming idea

Goal:

- avoid misdiagnosing a system bottleneck as an observer bottleneck
- cut the dominant remaining cost (repeated full-sentence decode) by
  giving the Gemma ASR path the same stateful-streaming contract that
  makes `qwen_forced` fast

Rationale — what the current code actually does and why it is slow:

- `CascadeSession.transcribe_audio` slices
  `state.source[utt_timestamps[-1]:]` and hands the whole growing
  audio tail to the backend every chunk
- `utt_timestamps[-1]` only advances on sentence-level punctuation
  commits, so between commits the backend re-decodes every previously
  generated word from scratch each chunk
- on the diagnostic single-audio run recorded earlier in this doc,
  decode was `2821 ms` of `3810 ms` total (~74%), while prompt forward
  was only `630 ms`, so the repeated decode is the lever, not the
  audio encoder

Reusable idea from Qwen3-ASR (`.venv-inference/.../qwen_asr/inference/qwen3_asr.py`):

- `Qwen3ASRModel.streaming_transcribe` feeds all audio seen so far,
  but injects the previously decoded text (minus a configurable
  `unfixed_token_num` rollback tail) as an assistant-side prompt
  prefix
- as a result, each chunk only needs to decode the small text *delta*
  (rolled-back tail + any new words), not the whole sentence again
- this matches vLLM's prefix-cache friendly serving pattern on the
  text side without requiring KV-cache-native observer work on the
  audio side

Concrete direction for Gemma:

- carry a per-utterance `streaming_prefix_text` through
  `AlignmentBackend.transcribe_and_align`
- `gemma_vllm_qk_fast` renders its chat prompt with that text as the
  assistant prefill so generation resumes after it
- a small rollback tail (a handful of tokens) keeps boundary
  corrections and punctuation-driven sentence commits possible
- state resets at sentence commits (`utt_timestamps` advance) so the
  next sentence starts cold

Scope boundaries (to honour "do not conflate observer work with
incremental-ASR work"):

- do not change observer semantics
- do not re-enable vLLM prefix caching as part of this step
- observer still captures only the newly generated token span each
  call; timings for the frozen prefix are carried over from the
  previous call, exactly as they already stabilised then

Required outcome:

- streaming prefix-prompt mode available behind an explicit flag on
  `gemma_vllm_qk_fast`
- `gemma_vllm_qk_fast` RTF on `tmp/alignatt_smoke18.wav` drops
  materially below `2.305` under that flag without regressing text
  quality vs the non-streaming path
- then the same check on `ccpXHNfaoy_short60s.wav` to confirm the gain
  is not smoke18-specific

Stop condition:

- backend-observer work and incremental-ASR work are tracked as
  separate questions with separate success criteria
- streaming mode is opt-in, not a silent default


## Gated Surface Change

The runtime surface now has three valid alignment backend names:

- `qwen_forced` — stable, default
- `gemma_onepass_qk_fast` — stable, Transformers-based Gemma path
- `gemma_vllm_qk_fast` — **experimental**, vLLM-based Gemma path

The following constraints remain:

- no replacement of `gemma_onepass_qk_fast`
- `gemma_vllm_qk_fast` is opt-in, not included in default comparisons
- no claim that the experimental vLLM backend is ready for SimulStream
- no multi-audio sweep before the mono-clip picture is stable

The vLLM backend remains:

- experimental
- mono-audio-first
- research-only until parity and robustness are clearer


## Minimal validation that must stay in scope

### Unit-level validation (32 tests pass)

The following invariants are covered by `test_alignment_helpers.py`:

- tensor observer buffers can round-trip to a compact payload
- out-of-span positions do not corrupt edge slots
- compact `qk_fast` reconstruction does not require a giant dump format
- observer module identity is reused on compatible reconfiguration
- prompt observer cache requires complete prompt capture
- prompt observer cache hydrates missing keys and ignores mismatches
- `reset_caches()` clears prompt observer cache and drift state
- decode-drift detection reports divergence index and token counts
- CLI accepts all compilation/repeat/seam-comparison arguments
- prefix caching is disabled by default for the observer path
- `gemma_vllm_qk_fast` is accepted by `CascadeRuntimeConfig` and is in
  `VALID_ALIGNMENT_BACKEND_NAMES` but not in `STABLE_ALIGNMENT_BACKEND_NAMES`
- vLLM runtime config defaults match the validated `cudagraph=full` seam

### System-level validation

The system loop is now automated via two CLI commands:

- `seam_comparison` runs the three-seam triplet on one audio:
  eager baseline, `cudagraph=full`, `vllm_compile + cudagraph=none`
- `gemma_vllm_inspect --repeat N` runs the same seam N times on the
  same engine and reports text stability and decode drift

Do not move to multi-audio sweeps before that set is successful and
interpretable.


## What remains explicitly out of scope

These are still non-goals for the current stage:

- rewriting MT
- adding transcript-cleanup heuristics
- exposing full attention dumps to the runtime
- claiming that vLLM alone solves the cascade
- conflating backend-observer work with prefix-rerun elimination


## Final reading

The version worth defending in a paper is no longer:

"Gemma under vLLM somehow."

It is:

**a multimodal causal LLM with an ASR-side AlignAtt observer that
survives real engine-native execution and returns only the compact
statistics required for monotone streaming acceptance**

The best current evidence says:

- that goal is now validated on one clip under both `cudagraph=full` and
  `vllm_compile + cudagraph=none`
- `cudagraph=full` is still the fastest validated seam on that clip
- repeated same-engine runs are also stable on that clip if prefix
  caching is disabled
- the next priority is hardening the bootstrap-based observer substrate,
  and deciding on a KV-cache-native prompt observer path, not returning
  to late Python-hook experiments
