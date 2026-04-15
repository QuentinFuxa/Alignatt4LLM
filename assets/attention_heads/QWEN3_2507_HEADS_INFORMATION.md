# Qwen3-4B-Instruct-2507 Token Alignment Heads

Date: 2026-04-13

## Scope

This note records the token alignment head detection results for:

- `Qwen/Qwen3-4B-Instruct-2507`
- directions: `en-zh`, `en-de`, `en-it`
- detector script: `src/assets/attention_heads/detect_translation_heads.py`
- alignment inputs reused from existing clean files:
  - `word_alignments_en-zh.json`
  - `word_alignments_en-de.json`
  - `word_alignments_en-it.json`

No OpenAI alignment re-annotation was rerun for this pass.

## Output Files

- `translation_heads_Qwen_Qwen3-4B-Instruct-2507_en-zh.json`
- `translation_heads_Qwen_Qwen3-4B-Instruct-2507_en-de.json`
- `translation_heads_Qwen_Qwen3-4B-Instruct-2507_en-it.json`
- `translation_heads_Qwen_Qwen3-4B-Instruct-2507_summary.json`

## Per-Direction Results

### en-zh

- used pairs: `880`
- used target tokens: `7726`
- token alignment heads: `107 / 1152`
- top heads:
  - `L15H9` TS=`0.798712`
  - `L5H4` TS=`0.789048`
  - `L12H24` TS=`0.786378`
  - `L19H27` TS=`0.736324`
  - `L15H11` TS=`0.735124`

### en-de

- used pairs: `903`
- used target tokens: `14372`
- token alignment heads: `91 / 1152`
- top heads:
  - `L15H9` TS=`0.773120`
  - `L15H11` TS=`0.653011`
  - `L12H24` TS=`0.625814`
  - `L9H16` TS=`0.623700`
  - `L14H12` TS=`0.617726`

### en-it

- used pairs: `907`
- used target tokens: `15353`
- token alignment heads: `90 / 1152`
- top heads:
  - `L15H9` TS=`0.812216`
  - `L9H16` TS=`0.687350`
  - `L12H24` TS=`0.686611`
  - `L15H11` TS=`0.676321`
  - `L14H12` TS=`0.641245`

## Cross-Direction Summary

- heads common to all 3 directions: `81`
- heads present in top-20 for all 3 directions: `14`

Most stable common heads across `en-zh`, `en-de`, `en-it`:

| Layer | Head | Avg TS | en-de rank/TS | en-it rank/TS | en-zh rank/TS |
|---|---:|---:|---|---|---|
| 15 | 9 | 0.794683 | `#1 / 0.773120` | `#1 / 0.812216` | `#1 / 0.798712` |
| 12 | 24 | 0.699601 | `#3 / 0.625814` | `#3 / 0.686611` | `#3 / 0.786378` |
| 15 | 11 | 0.688152 | `#2 / 0.653011` | `#4 / 0.676321` | `#5 / 0.735124` |
| 9 | 16 | 0.673252 | `#4 / 0.623700` | `#2 / 0.687350` | `#6 / 0.708705` |
| 5 | 4 | 0.663221 | `#8 / 0.576275` | `#8 / 0.624339` | `#2 / 0.789048` |
| 19 | 27 | 0.660650 | `#7 / 0.605721` | `#6 / 0.639904` | `#4 / 0.736324` |
| 15 | 8 | 0.626206 | `#6 / 0.610998` | `#7 / 0.635046` | `#8 / 0.632573` |
| 14 | 12 | 0.614316 | `#5 / 0.617726` | `#5 / 0.641245` | `#9 / 0.583977` |
| 13 | 29 | 0.591083 | `#10 / 0.529990` | `#9 / 0.572169` | `#7 / 0.671091` |
| 18 | 15 | 0.530947 | `#9 / 0.540917` | `#10 / 0.551242` | `#15 / 0.500681` |
| 13 | 10 | 0.522620 | `#11 / 0.528895` | `#11 / 0.527634` | `#14 / 0.511330` |
| 21 | 10 | 0.499740 | `#12 / 0.487039` | `#15 / 0.439207` | `#10 / 0.572974` |
| 13 | 9 | 0.480987 | `#13 / 0.478295` | `#12 / 0.497203` | `#17 / 0.467462` |
| 11 | 10 | 0.472619 | `#15 / 0.426615` | `#14 / 0.451787` | `#11 / 0.539455` |

Practical reading:

- `L15H9` is the strongest and most stable alignment head.
- The main cross-direction cluster is concentrated around layers `12-19`.
- The strongest reusable Qwen3 alignatt set should start from:
  - `L15H9`
  - `L12H24`
  - `L15H11`
  - `L9H16`
  - `L5H4`
  - `L19H27`
  - `L15H8`
  - `L14H12`

## Rerun Commands

Reuse existing cleaned alignments:

```bash
UV_PROJECT_ENVIRONMENT=.venv-inference uv run python src/assets/attention_heads/detect_translation_heads.py \
  --step detect \
  --direction en-zh \
  --mt-model 'Qwen/Qwen3-4B-Instruct-2507' \
  --alignment-file src/assets/attention_heads/word_alignments_en-zh.json \
  --device cuda:0 \
  --dtype bfloat16 \
  --output-dir src/assets/attention_heads
```

```bash
UV_PROJECT_ENVIRONMENT=.venv-inference uv run python src/assets/attention_heads/detect_translation_heads.py \
  --step detect \
  --direction en-de \
  --mt-model 'Qwen/Qwen3-4B-Instruct-2507' \
  --alignment-file src/assets/attention_heads/word_alignments_en-de.json \
  --device cuda:0 \
  --dtype bfloat16 \
  --output-dir src/assets/attention_heads
```

```bash
UV_PROJECT_ENVIRONMENT=.venv-inference uv run python src/assets/attention_heads/detect_translation_heads.py \
  --step detect \
  --direction en-it \
  --mt-model 'Qwen/Qwen3-4B-Instruct-2507' \
  --alignment-file src/assets/attention_heads/word_alignments_en-it.json \
  --device cuda:0 \
  --dtype bfloat16 \
  --output-dir src/assets/attention_heads
```

## Recommended Files To Use In Code

For direct runtime use per direction:

- `translation_heads_Qwen_Qwen3-4B-Instruct-2507_en-zh.json`
- `translation_heads_Qwen_Qwen3-4B-Instruct-2507_en-de.json`
- `translation_heads_Qwen_Qwen3-4B-Instruct-2507_en-it.json`

For manual inspection and cross-direction selection:

- `translation_heads_Qwen_Qwen3-4B-Instruct-2507_summary.json`
- `QWEN3_2507_HEADS_INFORMATION.md`
