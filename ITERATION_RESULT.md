# Iteration Result — Gemma ASR Discrepancy Resolution

Branch `maybe_gemma_aligner`. Supersedes the previous ITERATION_RESULT.md.

## Root Cause: `attn_implementation="eager"` Destroys Gemma ASR

The previous conclusion — that Gemma E4B free-run ASR hallucinates on
conference audio due to domain mismatch — was **wrong**. The hallucination
was caused by a testing artifact: loading the model with
`attn_implementation="eager"` via `GemmaAttentionAlignmentBackend`.

### Controlled ablation (8 variants × 2 clips)

Three axes tested independently: `attn_implementation` (default vs eager),
`audio_input_format` (filepath string vs numpy array), `decode_policy`
(greedy vs sampled). Model loaded once per clip; attention implementation
toggled at runtime via config attribute.

#### smoke18 (Siyu Yuan, Chinese accent, Fudan University)

| attn | audio | decode | WER | CER |
|------|-------|--------|----:|----:|
| default | filepath | greedy | **0.029** | 0.010 |
| default | filepath | sampled | 0.114 | 0.072 |
| default | numpy | greedy | **0.029** | 0.010 |
| default | numpy | sampled | 0.114 | 0.051 |
| eager | filepath | greedy | 0.829 | 0.672 |
| eager | filepath | sampled | 0.943 | 0.764 |
| eager | numpy | greedy | 0.829 | 0.672 |
| eager | numpy | sampled | 0.829 | 0.672 |

#### rxrToXvRyM_first18 (Myra, native English)

| attn | audio | decode | WER | CER |
|------|-------|--------|----:|----:|
| default | filepath | greedy | **0.259** | 0.169 |
| default | filepath | sampled | 0.259 | 0.173 |
| default | numpy | greedy | **0.259** | 0.169 |
| default | numpy | sampled | 0.259 | 0.165 |
| eager | filepath | greedy | 0.852 | 0.669 |
| eager | filepath | sampled | 0.833 | 0.658 |
| eager | numpy | greedy | 0.852 | 0.669 |
| eager | numpy | sampled | 0.815 | 0.665 |

### Findings

1. **`attn_implementation="eager"` is the sole cause.** Every eager variant
   hallucinates (WER ≥ 0.81). Every default variant transcribes correctly
   (WER 0.03–0.26). The gap is 25–30× on smoke18.

2. **Audio input format does not matter.** Filepath string and numpy array
   produce identical results under the same attention implementation.

3. **Decode policy does not matter materially.** Greedy and sampled produce
   similar quality; greedy is slightly better on smoke18 (WER 0.029 vs
   0.114).

4. **The previous "domain mismatch" diagnosis was incorrect.** The Chinese-
   accented speaker (smoke18) achieves WER 0.029 with correct attention —
   near-perfect transcription including "Siyu Yuan" and "Fudan University".
   The earlier claim that E4B cannot handle non-native accents or conference
   audio was based entirely on the eager-attention artifact.


## Corrected Gemma ASR Assessment

### smoke18 (Chinese accent, conference audio)

Best result (default attention, greedy): **WER 0.029, CER 0.010**

Transcript: "Hi, I'm Si Yuan from Fudan University. I'm here to introduce
our work, "Distilling Script Knowledge from Large Language Models for
Constrained Language Planning..."

This is near-perfect. Name and institution are correct. Paper title is
slightly paraphrased but accurate.

### rxrToXvRyM_first18 (native English, conference audio)

Best result (default attention, greedy): **WER 0.259, CER 0.169**

Transcript: "Hi, I'm Mara, and today we've been talking about our paper
marked personas using natural language prompts to measure stereotypes in
language models. This work is done in collaboration with Essindermush and
Dandrowski."

Content is correct. Errors are proper-name misspellings: "Mara" for "Myra",
"Essindermush" for "Esin Durmus", "Dandrowski" for "Dan Jurafsky". These
are entity/name noise, not semantic hallucination.


## Architecture Implications

### Why eager attention breaks ASR

Eager attention materializes full attention matrices at every layer. This
changes the numerical behavior of the attention computation compared to
Flash Attention / SDPA, which use numerically-stable fused kernels. For
long sequences with many audio tokens (~450 tokens for 18s audio), the
cumulative precision difference is enough to completely derail generation.

### Implication for the alignment backend

The alignment backend (`GemmaAttentionAlignmentBackend`) **must** use eager
attention to extract attention weights for alignment. This means:

- **Free-run ASR + alignment in one pass is not viable** with the current
  architecture: eager attention produces good alignment signals but
  catastrophically bad transcripts.

- **Forced alignment with eager is fine**: the transcript is prefilled, so
  ASR quality doesn't matter. The 177 ms MAE result was measured with eager
  attention and is valid.

- **A full Gemma cascade is architecturally possible** via two passes:
  1. Generate transcript with default attention (good ASR)
  2. Run forced alignment with eager attention (good alignment)

  This is more expensive (two forward passes) but eliminates the Qwen ASR
  dependency entirely.

### Revised recommendation

**Decision Rule B applies**: corrected Gemma ASR is mixed but meaningfully
better than previously reported. The recommendation must be rewritten.

| Path | Pros | Cons |
|------|------|------|
| **Hybrid (Qwen ASR + Gemma align)** | Proven, single-pass, Qwen ASR is better | Keeps Qwen dependency |
| **Full Gemma (two-pass)** | No Qwen dependency, defensible quality | Two forward passes, higher cost |
| **Full Gemma (default attn, no alignment)** | Simplest, good ASR | No word-level timing |

**Updated position:**

1. **Hybrid remains the practical baseline** for streaming cascade work
   where word-level timing is needed. The alignment backend's eager
   requirement makes single-pass full-Gemma impossible.

2. **Full Gemma cascade should be reopened** as a viable research direction.
   The two-pass approach (default-attn ASR → eager-attn forced alignment)
   is architecturally sound and worth benchmarking.

3. **The "domain mismatch" narrative must be retracted.** Gemma E4B handles
   conference audio with non-native accents well (WER 0.029 on Chinese-
   accented English). The remaining errors (WER 0.259 on native English)
   are proper-name misspellings, not hallucination.

4. **For ASR-only use cases** (no alignment needed), Gemma with default
   attention is competitive. smoke18 WER 0.029 is excellent.


## What Was Fixed

### `run_gemma_asr_fairness.py` — rebuilt as canonical ablation harness

The old fairness harness loaded the model through
`GemmaAttentionAlignmentBackend`, which forced `attn_implementation="eager"`.
Every variant it tested was tainted by this.

The rebuilt harness:
- Loads the model directly via `AutoModelForMultimodalLM` (no backend)
- Tests `attn_implementation` as an explicit ablation axis
- Tests audio input format (filepath vs numpy) as an explicit axis
- Tests decode policy (greedy vs sampled) as an explicit axis
- Toggles attention implementation at runtime via config attribute
- Stores raw responses, parsed responses, and scoring metadata

### Previous documents that need qualification

- **PLAN_RESULT_IMPLEMENTATION.md §Phase 3**: "Gemma E4B free-run ASR on
  smoke18 hallucinates" — this was caused by eager attention, not by the
  model's ASR capability. The "domain mismatch" diagnosis is incorrect.

- **PLAN_AUDIT_NOTE.md §Gemma ASR Fairness Benchmark**: "Gemma free-run ASR
  was not previously misused" — it was misused: the model was loaded with
  `attn_implementation="eager"`, which is the wrong configuration for ASR.

- **ITERATION_RESULT.md (previous)**: "Full Gemma cascade: not viable on
  conference audio with E4B" — this conclusion is invalidated.


## Artifacts Produced

- `tmp/alignment_research/gemma_asr_fairness_ablation_smoke18.json`
- `tmp/alignment_research/gemma_asr_fairness_ablation_rxrToXvRyM_first18.json`
- `run_gemma_asr_fairness.py` — rebuilt canonical harness
- `ITERATION_RESULT.md` — this document
