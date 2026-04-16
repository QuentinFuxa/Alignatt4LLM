# PLAN.md
# Next Agent Brief: Rebuild and Reassess Gemma ASR with the Exact Multimodal Path

## 1. Mission

Your job is to resolve a serious methodological problem in our current Gemma ASR story.

We previously concluded that Gemma free-run ASR was extremely poor on our conference clips.
A new standalone test, using the exact official multimodal inference path, produced much better results on the same local audio.

That means our previous negative conclusion is no longer defensible.

Your task is to:

1. explain the discrepancy rigorously
2. rebuild the Gemma ASR evaluation around the exact working path
3. reassess what architectural claims are actually justified
4. update the repo so future work uses the correct path by default

This is not a prompt-tuning task.
This is a research-integrity and systems-redesign task.


## 2. What Is Already Known

These points should be treated as current working facts unless you disprove them carefully.

### 2.1 Previous fairness conclusion is suspect

Earlier repo-level fairness runs suggested Gemma ASR was catastrophically bad, including on:

1. `tmp/alignatt_smoke18.wav`
2. `tmp/rxrToXvRyM_first18.wav`

That conclusion was used to justify a strong preference for hybrid over full Gemma ASR.

### 2.2 New standalone multimodal test gives much better results

A new script now exists:

- [standalone_gemma_asr_test.py](/home/fuxa/cascade_simultaneous/standalone_gemma_asr_test.py)

It uses the official-style Gemma multimodal pattern:

1. `AutoProcessor`
2. `AutoModelForMultimodalLM`
3. audio before text in the prompt
4. `processor.apply_chat_template(..., add_generation_prompt=True)`
5. `model.generate(...)`
6. `processor.decode(...)`
7. `processor.parse_response(...)`
8. standardized sampling:
   - `temperature=1.0`
   - `top_p=0.95`
   - `top_k=64`
   - `do_sample=True`

### 2.3 Observed standalone results

From:

- [standalone_gemma_asr_results.json](/home/fuxa/cascade_simultaneous/tmp/standalone_gemma_asr_results.json)

Observed outcomes:

1. Official sample audio transcribes correctly.
2. `tmp/alignatt_smoke18.wav` is much better than previously reported.
   - observed quick score: `WER ~= 0.0857`
   - observed quick score: `CER ~= 0.0462`
3. `tmp/rxrToXvRyM_first18.wav` is imperfect but not catastrophically wrong.
   - observed quick score: `WER ~= 0.2593`
   - observed quick score: `CER ~= 0.1618`

### 2.4 What this means

At minimum, one of the following is true:

1. the old fairness harness used Gemma incorrectly
2. the old harness and new standalone test are not actually evaluating the same thing
3. some implementation difference materially changes Gemma ASR quality

Until that is resolved, we must not make strong architectural claims from the old fairness results.


## 3. Why This Matters

The repo is trying to make a defensible research argument around:

1. Qwen ASR
2. Gemma attention-based alignment
3. hybrid and possibly full-Gemma cascade variants

If Gemma ASR was judged using the wrong runtime path, then:

1. the current hybrid-vs-full-Gemma recommendation may be biased
2. the current negative result may be invalid
3. the write-up risks making a claim we could not defend in a paper

Your work should make the Gemma ASR story clean enough that we can honestly say one of these:

1. Gemma ASR is viable on our target audio with the correct path
2. Gemma ASR is mixed but usable in some regimes
3. Gemma ASR is still not good enough, but now that conclusion is based on the correct evaluation path


## 4. Hard Constraints

You must follow these constraints.

### 4.1 Research integrity

1. Do not add lexical repairs.
2. Do not add content-aware substitutions.
3. Do not patch outputs with dataset-specific hacks.
4. Do not rescue a result with prompt tinkering that we could not defend in a paper.
5. Do not continue repeating the old negative conclusion unless it survives the corrected benchmark.

### 4.2 Cost discipline

1. Treat model loading and long streaming runs as expensive.
2. Do not restart hot models unless necessary.
3. Validate the mechanism on one or two clips before scaling.
4. Do not launch broad sweeps early.

### 4.3 Scope discipline

This iteration is about the Gemma ASR discrepancy first.

Not first priority:

1. new alignment-head discovery
2. new hybrid ablations
3. full cascade benchmarking
4. broad robustness studies

You may touch those only after the ASR-path discrepancy is resolved.


## 5. Files You Must Read First

Read these before making architecture claims or code changes.

### 5.1 Repo instructions and context

1. [AGENTS.md](/home/fuxa/cascade_simultaneous/AGENTS.md)
2. [CLAUDE.md](/home/fuxa/cascade_simultaneous/CLAUDE.md)

### 5.2 Recent result narratives

1. [PLAN_RESULT_IMPLEMENTATION.md](/home/fuxa/cascade_simultaneous/PLAN_RESULT_IMPLEMENTATION.md)
2. [PLAN_AUDIT_NOTE.md](/home/fuxa/cascade_simultaneous/PLAN_AUDIT_NOTE.md)
3. [ITERATION_RESULT.md](/home/fuxa/cascade_simultaneous/ITERATION_RESULT.md)

### 5.3 Code paths to inspect closely

1. [standalone_gemma_asr_test.py](/home/fuxa/cascade_simultaneous/standalone_gemma_asr_test.py)
2. [run_gemma_asr_fairness.py](/home/fuxa/cascade_simultaneous/run_gemma_asr_fairness.py)
3. [gemma_alignment_probe.py](/home/fuxa/cascade_simultaneous/gemma_alignment_probe.py)
4. [qwen3asr_gemma_cascade_core.py](/home/fuxa/cascade_simultaneous/qwen3asr_gemma_cascade_core.py)
5. [hybrid_alignment_backend.py](/home/fuxa/cascade_simultaneous/hybrid_alignment_backend.py)

### 5.4 Result artifacts to compare directly

1. [standalone_gemma_asr_results.json](/home/fuxa/cascade_simultaneous/tmp/standalone_gemma_asr_results.json)
2. [gemma_asr_fairness_smoke18.json](/home/fuxa/cascade_simultaneous/tmp/alignment_research/gemma_asr_fairness_smoke18.json)
3. [gemma_asr_fairness_rxrToXvRyM_first18.json](/home/fuxa/cascade_simultaneous/tmp/alignment_research/gemma_asr_fairness_rxrToXvRyM_first18.json)
4. [smoke18_reference.txt](/home/fuxa/cascade_simultaneous/tmp/alignment_research/smoke18_reference.txt)
5. [rxrToXvRyM_first18_reference.txt](/home/fuxa/cascade_simultaneous/tmp/rxrToXvRyM_first18_reference.txt)


## 6. Working Hypotheses to Test

You are expected to test these in a controlled way.

### H1. Audio input representation matters

Possible mismatch:

1. local file reference
2. remote URL reference
3. raw waveform array
4. preloaded audio tensors

It is plausible that the old fairness harness bypassed the path Gemma expects most naturally.

### H2. Decoding policy matters

Possible mismatch:

1. standard sampled generation
2. greedy generation
3. manual token-by-token argmax loop

If the old harness used a custom decode path, that may explain extreme quality loss.

### H3. Processor/model stack matters

Possible mismatch:

1. `AutoModelForMultimodalLM` vs other model classes
2. full processor stack available vs partially broken multimodal stack
3. missing dependency effects such as `torchvision`
4. dtype/device differences that alter behavior

### H4. Prompt and chat template path matter

Possible mismatch:

1. audio-first vs text-first ordering
2. `apply_chat_template` differences
3. generation prompt handling
4. parsing the returned assistant response correctly

### H5. Scoring path may have been wrong

Possible mismatch:

1. raw decoded output was scored incorrectly
2. special tokens or formatting noise were left in
3. reference mismatch or preprocessing mismatch inflated WER/CER


## 7. Primary Goal

Build one canonical Gemma ASR evaluation path that is:

1. official-style
2. reproducible
3. auditable
4. the default trustworthy benchmark in this repo

That path should replace or obsolete the misleading one.


## 8. Secondary Goal

After the corrected benchmark exists, decide which of these claims is justified:

1. full Gemma cascade should be reopened on current conference audio
2. hybrid remains the main path, but on corrected evidence
3. Gemma-only ASR should be limited to a cleaner-audio regime

This decision must come after the benchmark correction, not before.


## 9. Step-by-Step Work Plan

## Phase 1: Reproduce the discrepancy cleanly

### Objective

Show, with saved artifacts, that the old and new paths really disagree on the same clips.

### Tasks

1. Run the standalone multimodal script on:
   - official sample audio
   - `tmp/alignatt_smoke18.wav`
   - `tmp/rxrToXvRyM_first18.wav`
2. Run the old fairness harness on the same local clips.
3. Save outputs side by side.
4. Record:
   - prompt used
   - audio representation used
   - decoding config
   - raw response
   - parsed response
   - normalized transcript used for scoring
   - WER/CER

### Exit criterion

We have an artifact bundle that proves the mismatch directly and reproducibly.


## Phase 2: Root-cause the mismatch by controlled ablation

### Objective

Identify which implementation differences materially caused the earlier pessimistic result.

### Required ablation axes

#### A. Audio representation

Compare at least:

1. local file path string
2. remote URL if stable and available
3. raw waveform array

#### B. Decoding policy

Compare at least:

1. official sampled generation
2. greedy generation
3. any custom manual loop used in the earlier path

#### C. Model/processor path

Compare at least:

1. `AutoModelForMultimodalLM`
2. any old model-loading path still in the repo
3. exact processor setup and dependency assumptions

#### D. Response extraction and scoring

Compare at least:

1. raw decoded response
2. parsed assistant text
3. normalized transcript used for scoring

### Deliverable for this phase

Write a concise discrepancy report that states:

1. which factors matter
2. which factors do not matter
3. the most likely root cause of the old bad result
4. what the canonical path must be going forward

### Exit criterion

You can explain, with evidence, why the old fairness harness looked too bad.


## Phase 3: Rebuild the repo fairness harness

### Objective

Update the repo so future Gemma ASR evaluation uses the correct canonical path.

### Tasks

1. Update or replace [run_gemma_asr_fairness.py](/home/fuxa/cascade_simultaneous/run_gemma_asr_fairness.py).
2. Make the default code path match the working standalone method.
3. Preserve optional switches for controlled ablations, but keep the default path canonical.
4. Ensure outputs store enough metadata to audit every run.

### Canonical-path requirements

The corrected harness should, by default:

1. use `AutoProcessor`
2. use `AutoModelForMultimodalLM`
3. place audio before text
4. use `apply_chat_template(..., add_generation_prompt=True)`
5. use the standardized sampling config
6. store both raw response and parsed response
7. record transcript normalization used for scoring
8. record model id or snapshot path
9. record device/dtype information when available

### Exit criterion

There is one clear Gemma ASR benchmark path in the repo, and the old misleading path is removed or explicitly marked obsolete.


## Phase 4: Re-evaluate Gemma ASR on the current clip set

### Objective

Produce the corrected Gemma ASR story on the clips we already care about.

### Tasks

1. Re-run `tmp/alignatt_smoke18.wav`.
2. Re-run `tmp/rxrToXvRyM_first18.wav`.
3. If cost is reasonable, run one or two additional short conference clips.
4. Produce a table containing:
   - clip id
   - transcript output
   - WER
   - CER
   - notable error type

### Important analysis requirement

Separate these failure modes explicitly:

1. mostly correct transcript with some entity/name noise
2. fluent but semantically wrong hallucination
3. general acoustic failure

That distinction matters for the architecture decision.

### Exit criterion

We have a corrected, credible Gemma ASR assessment on the clips already under discussion.


## Phase 5: Reassess the architecture story

### Objective

Update the repo’s recommendation based on the corrected ASR evidence.

### Questions to answer

1. Is full Gemma ASR now viable enough to deserve renewed work on these conference clips?
2. Is hybrid still the best practical path?
3. Should the story be split by domain, for example:
   - hard conference audio -> Qwen ASR or hybrid
   - cleaner audio -> possible full Gemma path
4. Does the standalone Gemma aligner remain the core contribution regardless of ASR outcome?

### Required output

Produce a short written recommendation that says clearly which of these is now the most defensible claim:

1. reopen full Gemma cascade on current audio
2. keep hybrid as baseline but revise the write-up
3. restrict full Gemma cascade exploration to a cleaner regime

### Exit criterion

The repo has one updated architecture recommendation that reflects corrected Gemma ASR evidence.


## 10. Non-Goals for This Iteration

Unless the ASR discrepancy is already resolved, do not spend significant time on:

1. discovering new Gemma alignment heads
2. retuning hybrid fallback policy
3. full streaming evaluation campaigns
4. broad multi-audio benchmark sweeps
5. writing paper prose beyond concise result notes


## 11. Expected Deliverables

By the end of this iteration, produce all of the following.

### Code deliverables

1. corrected or replaced [run_gemma_asr_fairness.py](/home/fuxa/cascade_simultaneous/run_gemma_asr_fairness.py)
2. any supporting helper changes needed to make the canonical path clean and reusable

### Artifact deliverables

1. side-by-side discrepancy artifacts
2. corrected fairness outputs on the current clip set
3. a concise discrepancy report
4. a concise architecture recommendation update

### Documentation deliverables

Update whichever note is most appropriate among:

1. [PLAN_RESULT_IMPLEMENTATION.md](/home/fuxa/cascade_simultaneous/PLAN_RESULT_IMPLEMENTATION.md)
2. [PLAN_AUDIT_NOTE.md](/home/fuxa/cascade_simultaneous/PLAN_AUDIT_NOTE.md)
3. [ITERATION_RESULT.md](/home/fuxa/cascade_simultaneous/ITERATION_RESULT.md)

If none of them is appropriate, create one short new result note rather than scattering conclusions across many files.


## 12. Decision Rules

Use these rules when deciding whether the repo story should change.

### Rule A

If corrected Gemma ASR is clearly good on the main conference clips, do not keep the old negative conclusion.
Reopen the full Gemma cascade path honestly.

### Rule B

If corrected Gemma ASR is mixed but meaningfully better than previously reported, rewrite the recommendation to reflect that nuance.
Do not overstate hybrid necessity.

### Rule C

If corrected Gemma ASR is still weak, but now weak under the correct path, then the negative result is valid again.
At that point hybrid remains the main path on defensible grounds.

### Rule D

If the answer varies sharply by audio type, say so explicitly.
A domain-split recommendation is acceptable if it is supported by evidence.


## 13. Suggested Execution Order

If you want a minimal and disciplined path, do this in order:

1. inspect old fairness code against the standalone script
2. reproduce the mismatch on `smoke18`
3. isolate the dominant cause on `smoke18`
4. patch or replace the fairness harness
5. rerun `rxrToXvRyM_first18`
6. write the corrected architecture recommendation

Do not fan out early.


## 14. Final Standard

At the end of this iteration, another researcher should be able to look at the repo and understand:

1. why the earlier Gemma ASR result was misleading
2. what the correct Gemma ASR evaluation path is
3. what Gemma ASR can actually do on our current audio
4. whether hybrid vs full-Gemma claims are still justified

That is the bar.
