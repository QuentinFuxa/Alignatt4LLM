# `docs/archive/`

Historical design docs, iteration notes, and one-off plans preserved for context. **Not a source of current truth.** Always prefer the top-level `docs/` for the present state of the system.

Some of these docs reference paths like `assets/alignatt_doc/...` that no longer exist — the content in question has moved to `docs/reference/` or elsewhere. Internal references inside archive files are intentionally left as they were written, to preserve the original reading.

## Contents

- `E4B_ALIGNATT_CASCADE_DESIGN.md` — long-running design + iteration notes from the pre-`full_vllm`-merge period. Covers source-frontier, AlignAtt-rewind, provenance-aware acceptance, the en→it / en→zh multilingual validation, and the research-to-SimulStream delivery story. ~1700 lines.
- `ALIGNATT_LLM.md` — earlier theoretical write-up of AlignAtt applied to causal LLMs.
- `SIMULSTREAM_TWO_FRONTENDS.md` — snapshot from when the repo had two ASR frontends; superseded by `docs/RUNTIME_ARCHITECTURE.md`.
- `PLAN_test_cs_en_qwen3_gemma.md` — specific 3-audio en→cs test plan (partial blocker: `cs.txt` reference still missing).
- `PLAN.md`, `PLAN_AUDIT_NOTE.md`, `PLAN_RESULT_IMPLEMENTATION.md`, `ITERATION_RESULT.md` — earlier iteration PLAN cycle.
- `PLAN_qk_fast_audio.md`, `QK_FAST_AUDIO_IMPLEMENTATION_NOTE.md` — qk_fast audio validation history.

If you're looking for the **current** plan, read the root `PLAN.md` and `DECISIONS.md`.
