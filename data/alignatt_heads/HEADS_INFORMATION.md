# AlignAtt Head Artifacts

This directory keeps measured AlignAtt head artifacts. Text-side MT defaults are
Gemma-specific for the stable baseline cascade; MiLMMT files are active
post-submission research artifacts.

The main text-side artifact here is [`detect_translation_heads.py`](detect_translation_heads.py), which implements the paper-style Translation Score pipeline for causal decoder attention heads:

1. load parallel sentence pairs for a direction,
2. annotate word alignments with `gpt-5-mini`,
3. filter to reliable lexical anchors,
4. run the translation model with `output_attentions=True`,
5. score every `(layer, head)` by full-sequence argmax accuracy on aligned target tokens.

## What Lives Here

- `word_alignments_<direction>.json`
  Clean GPT-derived word mappings after anchor filtering.
- `raw_alignments_<direction>.jsonl`
  Resume-friendly raw alignment annotations.
- `translation_heads_xiaomi-research_MiLMMT-46-4B-v0_1_<direction>.json`
  Experimental ranked text translation heads for MiLMMT.
- `translation_heads_google_gemma-4-E4B-it_<direction>.json`
  Ranked text translation heads for the stable Gemma baseline route.
- `audio_alignment_heads_google_gemma-4-E4B-it_*.json`
  Audio-side attention-head bundles used by the current alignment probe.

## Current Defaults

- Translation model default: `google/gemma-4-E4B-it`
- Alignment model default: `gpt-5-mini`
- Direction default: `cs-en`

The detector no longer bakes in a local corpus path. You must pass explicit parallel text with `--src-path` and `--tgt-path` so we do not silently calibrate on stale or evaluation data.

## Czech -> English For IWSLT 2026

Use parallel Czech-English training text to discover heads, then validate the promoted head set on the official dev set.

Recommended order:

1. `CzEng 2.0` as the primary head-discovery corpus.
2. `Europarl` and `VoxPopuli cs->en translated data` as political / speech-domain supplements.
3. `OpenSubtitles v2018` only as an optional robustness supplement, not the core signal.

Do not use raw ASR-only corpora such as `ParCzech` or `Common Voice` by themselves for text-side head discovery, because the scoring algorithm needs parallel source-target text.

Do not use the official Czech-to-English test set for calibration.

Keep the official 2026 Czech dev set for downstream validation and promotion decisions.

More detail lives in [`IWSLT26_CZ_EN.md`](IWSLT26_CZ_EN.md).

## Current Czech -> English Artifact

The current `cs-en` text-head artifact is:

- [`translation_heads_google_gemma-4-E4B-it_cs-en.json`](translation_heads_google_gemma-4-E4B-it_cs-en.json)

It was scored from:

- `raw_alignments_cs-en.jsonl` (not vendored in this public branch)
- [`word_alignments_cs-en.json`](word_alignments_cs-en.json)

Run summary:

- raw aligned pairs collected: `516`
- filtered usable pairs: `512`
- aligned target tokens scored: `6043`
- token alignment heads found: `69 / 336`

Top `cs-en` heads:

1. `(11, 3)` with `TS=0.8341`
2. `(17, 3)` with `TS=0.7785`
3. `(20, 0)` with `TS=0.7369`
4. `(6, 5)` with `TS=0.6948`
5. `(11, 2)` with `TS=0.5755`
6. `(22, 4)` with `TS=0.5137`
7. `(7, 2)` with `TS=0.5037`
8. `(10, 0)` with `TS=0.4990`

## Cross-Language Comparison

Compared with the existing Gemma text-head runs:

- `en-de`: `84` heads from `903` scored pairs
- `en-it`: `83` heads from `907` scored pairs
- `en-zh`: `82` heads from `880` scored pairs
- `cs-en`: `69` heads from `512` scored pairs

The lower raw head count for `cs-en` should be interpreted mainly as a sample-size effect. The previous directions were scored on roughly `900` usable pairs, while the current Czech-English pass used `512`.

### Shared Core

The top-8 overlap of `cs-en` with every existing direction is `6 / 8`. The six heads shared by all four directions in the top-8 are:

- `(11, 3)`
- `(6, 5)`
- `(17, 3)`
- `(20, 0)`
- `(11, 2)`
- `(10, 0)`

This is the main qualitative result: Czech-English does not reveal a different alignment-head family. It strongly confirms the same multilingual Gemma alignment core already seen in `en-de`, `en-it`, and `en-zh`.

### Relative Similarity

At top-16 resolution, `cs-en` is closest to the European directions:

- overlap with `en-it`: `15 / 16`
- overlap with `en-de`: `14 / 16`
- overlap with `en-zh`: `12 / 16`

This suggests that Czech-English behaves more like the European directions than like English-Chinese, but the difference is modest enough that it should not be overclaimed from the current sample size.

### Consequence For The Shared Kernel

The existing shared-kernel artifact:

- [`translation_heads_shared_kernel_top8.json`](translation_heads_shared_kernel_top8.json)

contains `7` heads derived from `en-de`, `en-it`, and `en-zh`. Czech-English contains all of those heads as well, but one of them, `(5, 0)`, drops to rank `12` in `cs-en` rather than staying in the Czech top-8.

So:

- a strict top-8 intersection across all four directions would reduce the shared kernel from `7` heads to `6`
- keeping the current `7`-head shared kernel remains a defensible baseline
- if Czech-English is folded into the multilingual regime, a more stable choice is a shared kernel built from the four-language top-16 intersection

That four-language top-16 shared kernel contains `12` heads:

- `(11, 3)`
- `(6, 5)`
- `(17, 3)`
- `(20, 0)`
- `(11, 2)`
- `(5, 0)`
- `(10, 0)`
- `(7, 2)`
- `(10, 4)`
- `(22, 4)`
- `(6, 4)`
- `(17, 0)`

## Example

```bash
python data/alignatt_heads/detect_translation_heads.py \
  --direction cs-en \
  --model google/gemma-4-E4B-it \
  --src-path /path/to/czeng.cs \
  --tgt-path /path/to/czeng.en \
  --dataset-name czeng2.0 \
  --workers 20
```
