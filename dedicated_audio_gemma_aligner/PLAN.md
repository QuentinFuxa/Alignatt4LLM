# Dedicated Audio Gemma Aligner Plan

## Goal
Keep the trained Gemma-audio-feature aligner as a self-contained research track with clean code, stable artifact paths, and honest evaluation.

## What lives here
- The frozen-Gemma feature extractor
- The learned transcript-conditioned aligner head
- The split builders and teacher-generation utilities
- The saved checkpoints, manifests, teacher artifacts, and result notes

## Current scientific status
- The v2 result established that the tiny proof-of-concept fit but did not generalize.
- The v3 story is not fully clean yet: the markdown description and the checked-in trainer do not perfectly match.
- This track is still worth keeping because it is the main learned-aligner alternative to raw-head audio AlignAtt.

## Immediate priorities
1. Provenance cleanup
- Keep the experiment note synchronized with the exact checked-in trainer.
- Either implement a true hard-discrete v3a path or stop describing it as if it already exists in-repo.

2. Honest evaluation
- Treat `artifacts/feature_aligner/heldout_eval_v2.json` as the only fully audited held-out result today.
- Do not make strong representation-level claims from `training_summary_v3.json` alone.
- Save a complete v3 held-out evaluation artifact before drawing new conclusions.

3. Architectural triage
- If this frozen-feature path stays weak on held-out ACL data, keep it as a documented negative/partial result.
- Prefer `qk_fast` audio AlignAtt under `sdpa` as the primary non-Qwen research direction if that path works.

4. Repo hygiene
- Keep all generated artifacts under `artifacts/feature_aligner/`.
- Keep root-level wrappers minimal and avoid spreading new aligner-specific files back into the repo root.

## Success condition
A new contributor should be able to find the entire dedicated-audio-aligner story here, understand what is solid versus incomplete, and continue the work without hunting through the repo root.
