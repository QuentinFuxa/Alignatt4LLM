# Generalizing AlignAtt4LLM to Other LLMs

AlignAtt4LLM is not tied to the ASR system used for IWSLT. The ASR side only
supplies a growing source transcript plus timestamps. The portable part is the
MT-side policy: draft with a decoder-only LLM, recover where each drafted target
token attends in the source prompt, and commit only the safe target prefix.

> For the concrete, step-by-step recipe and a worked example, see
> [Adding a New LLM](adding_a_model.md) and the Qwen3 reference backend
> (`src/alignatt4llm/mt/qwen_vllm_backend.py`, `qwen_vllm_alignatt`). The
> generic, reusable capture machinery lives in `src/alignatt4llm/vllm_qk/`.
> This document is the conceptual companion to that recipe.

## What Must Transfer

To port AlignAtt4LLM to another decoder-only MT model, the new backend needs to
provide four pieces.

1. A source-visible prompt

   The prompt must contain the current source text in a region that can be
   mapped back to prompt token positions. In the current code this is represented
   by `RenderedTranslationPrompt`, `PromptSourceMap`, and
   `PromptSourceUnitSpan` in `src/alignatt4llm/mt/base.py`.

2. Calibrated translation heads

   AlignAtt needs a ranked list of decoder attention heads that behave like
   source-target alignment heads for the model and language direction. The
   runtime loads these from JSON files in `data/alignatt_heads/`, where each
   item has `layer`, `head`, and `ts` fields. The discovery pipeline lives in
   `data/alignatt_heads/detect_translation_heads.py`.

3. Attention reconstruction

   For each drafted target token, the backend must recover attention rows from
   the selected target-side query heads to the source-token positions in the
   prompt. The production vLLM path captures prompt K and decode Q/K tensors,
   then reconstructs the source block needed by the policy. See
   `src/alignatt4llm/mt/gemma_vllm_observer.py`.

4. The common acceptance policy

   Once the backend has source-attention rows, it can reuse
   `AlignAttDecoderPolicy` from `src/alignatt4llm/mt/base.py`. The policy turns
   reconstructed attention into source positions, compares them with the
   accessible source frontier, and returns the target prefix that can be safely
   emitted.

## Porting Path

Start with a Transformers prototype if possible. Use `output_attentions=True`
to verify that the model has useful source-alignment heads and that the prompt
mapping is correct. This is slower, but it is the simplest way to debug the
idea before touching the vLLM observer path.

Then add a production backend:

1. Create an MT backend implementing `BaseMTBackend`.
2. Register it in `build_mt_backend()` and add its name to
   `VALID_MT_BACKEND_NAMES` in `src/alignatt4llm/runtime.py`.
3. Add a model resolver in `mt_model_name_for_backend()`.
4. Add a heads-path convention in `alignatt_heads_path_for()`.
5. Implement prompt rendering and source-token mapping for the model tokenizer.
6. Bind an observer to the model's decoder layers so prompt K and decode Q/K
   are captured for the selected heads.
7. Reuse `AlignAttDecoderPolicy` for partial decoding and keep the final
   `is_partial=False` path as a normal full translation.

Steps 1, 2, 4, and 6 are now driven by the generic base in
`src/alignatt4llm/vllm_qk/`: a backend supplies a `VLLMAttentionSpec` (which
attention class to patch + how its `forward` recomputes Q/K) and a thin
`BaseQKObserverWorker` subclass, rather than hand-writing the patch and worker.
See [Adding a New LLM](adding_a_model.md) for the exact edits.

The current examples are:

- `GemmaVLLMMTBackend`: Gemma-family chat-template backend used by the stable
  IWSLT MT route (keeps its own QK-norm-aware forward).
- `MiLMMTVLLMMTBackend`: experimental MiLMMT route that reuses the same vLLM
  Q/K observer mechanics with a raw translation prompt.
- `QwenVLLMMTBackend`: the reference "bring your own LLM" backend (Qwen3),
  built on the generic `vllm_qk` base with the standard forward (QK-norm on).

## Model Requirements

A good candidate LLM should have:

- a tokenizer that can provide offsets, or another reliable way to map the
  source substring to prompt token positions;
- decoder layers and attention projections that can be inspected or patched;
- deterministic decoding support for the draft path;
- enough context budget for the source prompt, optional history, and draft;
- translation ability for the target directions being streamed.

Grouped-query or multi-query attention is usable, but the backend must map
query heads to their corresponding KV heads. The helper
`map_attention_head_to_key_value_head()` in `src/alignatt4llm/mt/base.py`
handles the current mapping logic.

## Head Calibration

Head artifacts are model- and direction-specific. Do not reuse Gemma heads for
a new model unless you are explicitly running a diagnostic. The intended flow
is:

1. collect legal parallel text for the source-target direction;
2. annotate word alignments;
3. score every decoder `(layer, head)` by translation score;
4. save `translation_heads_<model>_<direction>.json`;
5. validate the selected top-k heads on held-out development audio/text.

The detector intentionally requires explicit `--src-path` and `--tgt-path`
arguments so head discovery does not accidentally use evaluation data.

## What Does Not Need To Change

The ASR backend does not need to change when porting AlignAtt4LLM to another MT
LLM. Any upstream component can be used if it supplies:

- committed source text;
- source units, usually words or tokens;
- a timestamp/accessibility frontier for those units.

For a text-only streaming experiment, the same MT policy can run without ASR by
constructing the source frontier directly from text chunks.
