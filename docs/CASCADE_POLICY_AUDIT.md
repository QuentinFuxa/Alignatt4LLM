# Cascade Policy Audit

Snapshot date: 2026-04-17

This note records the **shipped** ASR->MT contract after re-reading
`scripts/legacy_baseline.py` carefully and deciding what the current cascade should own as
its explicit policy.

## Bottom line

The intended contract is now:

1. Keep `punctuation_lcp + EOS flush` on the ASR side to finalize sentence
   history and keep the prompt bounded.
2. On every partial MT update, feed MT the **full live ASR tail** for the
   current sentence.
3. Before building the MT prompt, strip only unstable trailing
   sentence-final punctuation from that live tail.
4. Let **MT-side AlignAtt alone** decide how much new target text is safe to
   accept and emit.

This is the clean separation we want:

- ASR punctuation commit controls **history management**
- AlignAtt controls **target emission**

## What `scripts/legacy_baseline.py` actually does

The baseline is easy to summarize too loosely as "local agreement".
Re-reading the code shows a more specific story:

1. **ASR sentence commit** by punctuation on the longest common prefix of two
   consecutive ASR hypotheses.
2. **Partial MT every round** on the full current ASR hypothesis.
3. **Target emission** by `longest_common_prefix(...)` between consecutive MT
   hypotheses.

Important subtlety: that MT-side local agreement is a **character-level**
target-target agreement rule. It is not an explicit source-grounding rule.

## Why the current cascade should not imitate that literally

The vLLM MT path already has a cleaner acceptance mechanism than the baseline:

- accepted target prefix is represented explicitly
- the prompt continues from that accepted prefix via assistant prefill
- AlignAtt reconstructs target->source links for each newly drafted token
- the runtime trims to the last target stability unit and stays append-only

So we do **not** need to copy the baseline's target-target local agreement.
The current MT path already has a stronger, more interpretable contract.

## The key design choice

The question was whether ASR stability should also limit the partial MT call.
The chosen answer is **no**.

We intentionally allow:

- MT to see the whole current ASR tail
- AlignAtt to be the sole mechanism that limits emitted target growth

That makes the policy easier to state honestly:

- the source side decides what is finalized history
- the target side decides what is safe to emit

We are **not** trying to build an ASR-stability frontier and then force MT
acceptance to stay behind it.

## What `punctuation_lcp` still does

`punctuation_lcp` remains important, but only for the source-side book-keeping:

- move finalized source text into committed sentence history
- expose stable sentence pairs as MT context
- flush the last sentence tail at EOS
- prevent the live tail from growing without bound

It is **not** the mechanism that limits partial MT emission.

## What AlignAtt now means in the policy story

AlignAtt should be read as:

- "Given the full MT-visible source prefix, which newly drafted target tokens
  are sufficiently grounded in the currently available source?"

That is the right role for it in this cascade:

- source conditioning stays rich
- target acceptance stays explicit and token-level
- the public stream remains append-only

## Practical implication for latency tuning

With this contract, the main latency knob is still `chunk_ms`, not some hidden
ASR stability threshold.

## Code-level consequences

The active runtime now states this contract directly:

- the live ASR tail normalization helper documents that it strips unstable
  trailing sentence-final punctuation and then hands the full tail to MT
- the source frontier builder documents that it operates over the full
  MT-visible source prefix and only gates target acceptance, not source
  conditioning
- the SimulStream processor no longer suppresses a partial MT update merely
  because the public ASR string repeated; source-frontier advancement alone
  may unlock new append-only target emission
- docs no longer suggest that we are trying to derive a separate ASR-stable
  frontier for MT acceptance

## Paper-friendly phrasing

If we describe the system in a paper, the honest version is:

> The MT model conditions on the full current ASR prefix for the ongoing
> sentence, after removing unstable trailing sentence-final punctuation. The
> runtime then uses MT-side AlignAtt to accept only the grounded prefix of each
> partial translation draft, yielding append-only target emission.

That is both cleaner and more defensible than saying we copied the baseline's
local-agreement policy.
