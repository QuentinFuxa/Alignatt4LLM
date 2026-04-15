# Token Alignment Heads in HyMT1.8B (EN-ZH)

## What Are Token Alignment Heads?

Token alignment heads are specific attention heads in a translation LLM that are responsible for mapping source language tokens to target language tokens during inference. They were identified by the paper "Token Alignment Heads: Unveiling Attention's Role in LLM Multilingual Translation" (ICLR 2026 submission).

When these heads are ablated (zeroed out), the model loses its ability to translate and reverts to copying the source text verbatim. Other heads can be removed with minimal impact on translation quality.

## Procedure

### Step 1: Word Alignment Annotation (GPT-5-mini)

We used **GPT-5-mini** (OpenAI) with structured JSON output to annotate word-level alignments between 919 English-Chinese parallel sentence pairs from the **MCIF dataset** (conference talk transcripts).

For each sentence pair, GPT-5-mini:
- Tokenized both source (English) and target (Chinese) into words
- Produced span-level alignments with confidence scores
- Example: `"Sara Papi"` <-> `"Sara Papi"` (conf=1.0), `"University of Trento"` <-> `"特伦托 大学"` (conf=0.9)

Result: **919 pairs annotated, 8450 raw alignments**.

### Step 2: Anchor Filtering

Not all alignments are useful for head detection. We filtered to keep only **high-quality anchor alignments** -- named entities, technical terms, numbers, and content words that serve as reliable position markers. We reject:
- Function words (the, a, is, etc.)
- Weak anchors (common verbs, pronouns)
- Pure punctuation
- Very short tokens without distinguishing signals (digits, uppercase, Greek letters)

Result: **4181 filtered alignments across 880 pairs** (49% keep rate).

### Step 3: Translation Score Computation

For each of the 880 pairs:
1. Build the HyMT translation prompt: `将以下文本翻译为中文...` + source + placeholder
2. Concatenate the reference target text
3. Tokenize and map word-level alignment spans to tokenizer token positions
4. Run a forward pass through HyMT1.8B with `output_attentions=True`
5. For each attention head at each layer, check:
   - For each target token that has a valid alignment, does the **full-sequence argmax** of that head's attention land on the aligned source token?
   - `TS_h = (# correct alignments) / (# total aligned target tokens)`

This is the exact algorithm from the paper (Section 2.2). The **full-sequence argmax** is critical -- we check whether the head's maximum attention across ALL positions (prompt, source, and previously generated target tokens) falls on the correct source token. This is much stricter than restricting the argmax to source-only positions.

### Step 4: Head Classification

A head is classified as a **token alignment head** if its average Translation Score across all 880 pairs exceeds **0.1** (paper's threshold).

## Results

**Model**: `tencent/HY-MT1.5-1.8B` (32 layers x 16 heads = 512 total)
**Direction**: English -> Chinese
**Data**: 880 MCIF pairs, 6769 scored target tokens

### Token Alignment Heads Found: 49 / 512 (9.6%)

| Rank | Layer | Head | Translation Score |
|------|-------|------|-------------------|
| 1 | 9 | 5 | 0.899 |
| 2 | 13 | 1 | 0.812 |
| 3 | 4 | 12 | 0.746 |
| 4 | 9 | 6 | 0.729 |
| 5 | 8 | 8 | 0.699 |
| 6 | 8 | 6 | 0.685 |
| 7 | 1 | 10 | 0.584 |
| 8 | 9 | 7 | 0.568 |
| 9 | 12 | 11 | 0.482 |
| 10 | 8 | 0 | 0.481 |

The top head (**Layer 9, Head 5**) correctly attends to the aligned source token **90% of the time** across 880 sentence pairs.

### Key Properties Observed

1. **Sparsity**: Only 9.6% of heads are token alignment heads. This is slightly above the paper's 3-8% range for general LLMs, which makes sense since HyMT is a translation-specialized model.

2. **Middle-layer concentration**: Token alignment heads are concentrated in layers 2-19, peaking at layers 8-15. The earliest and latest layers contain very few. This matches the paper's findings and the general understanding that middle layers handle cross-lingual mapping while early layers handle surface features and late layers structure the output.

3. **Top-heavy distribution**: The top 6 heads (layers 4, 8, 9, 13) account for the bulk of translation alignment capability, with TS > 0.68. There is then a sharp drop-off.

## Comparison with Previous Results (SimAlign-based)

We previously detected heads using SimAlign (unsupervised cross-lingual embeddings) with a **source-restricted argmax** -- only checking which source token gets maximum attention among source positions. That method found 16 heads.

The new method (GPT-5-mini + full-sequence argmax) reveals that **11 of those 16 old heads were false positives**. They appeared to attend to source tokens only because the argmax was restricted to source positions. In reality, their full-sequence attention maximum was on BOS or prompt tokens, not source tokens at all. Examples:

- L11H11: old TS=0.77, new TS=0.006 (attention was on BOS)
- L8H11: old TS=0.76, new TS=0.000
- L11H10: old TS=0.82, new TS=0.013

Only 5 of the 16 old heads survive the correct full-sequence test.

## How to Use These Heads

### 1. Alignment Attention (alignatt) for Streaming Translation

The primary use in this codebase is **alignment attention** -- using token alignment heads to determine WHEN to emit translation output in a simultaneous/streaming setting. During streaming translation:

- Monitor the top-K alignment heads' attention patterns
- When these heads shift attention from old source tokens to new ones, it signals that the model is now translating new content
- This provides a principled way to segment translation output without arbitrary time-based or token-count heuristics

The heads JSON is loaded by the cascade system in `systems/cascade_speech_processors/mt/alignatt_mt.py`.

### 2. Head Pruning / Ablation

Zeroing out token alignment heads collapses translation capability. This can be used for:
- **Controlled degradation studies**: Understanding how translation quality degrades as heads are removed
- **Model compression**: Identifying which heads MUST be preserved for translation

### 3. Training Data Filtering (TRater)

The paper introduces TRater: score training data by computing the loss difference when alignment heads are masked vs. unmasked. High-scoring data is critical for translation capability. This can guide data selection for fine-tuning.

## Files

| File | Description |
|------|-------------|
| `detect_translation_heads.py` | Full pipeline script (align + detect). Reusable on other models/languages. |
| `translation_heads_tencent_HY-MT1_5-1_8B_en-zh.json` | Head scores, ranked heads, and full 32x16 TS matrix |
| `word_alignments_en-zh.json` | Clean EN-ZH word mappings (880 pairs, 4181 alignments) |
| `raw_alignments_en-zh.jsonl` | Raw GPT-5-mini annotations (919 pairs, all confidence levels) |

## Rerunning on Other Models / Languages

```bash
# Different language pair with custom parallel data:
python detect_translation_heads.py --direction en-de \
  --src-path /path/to/en.txt --tgt-path /path/to/de.txt

# Different translation model:
python detect_translation_heads.py --mt-model tencent/HY-MT1.5-7B

# Use a different alignment LLM:
python detect_translation_heads.py --alignment-model gpt-4o-mini

# Skip alignment (reuse existing), only re-detect:
python detect_translation_heads.py --step detect \
  --alignment-file raw_alignments_en-zh.jsonl
```

Requirements: `transformers`, `torch` (GPU), OpenAI API key (`~/.openai_api_key` or `OPENAI_API_KEY` env var).
