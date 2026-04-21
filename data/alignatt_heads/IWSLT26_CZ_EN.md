# IWSLT 2026 Czech -> English Head Discovery

The attention-head detector in this repo is a text-side translation probe, so the right calibration data for `cs-en` is parallel Czech-English text, not raw Czech ASR audio by itself.

## What To Use

Primary corpus:

- `CzEng 2.0`

Best supplements:

- `Europarl`
- `VoxPopuli` translated `cs -> en` data

Optional robustness supplement:

- `OpenSubtitles v2018`

## What Not To Use As The Core Discovery Signal

- `ParCzech 3.0 (ASR)` alone
- `Common Voice Czech ASR` alone
- `MOSEL` transcripts alone

Those are useful elsewhere in the cascade, but they are not parallel Czech-English supervision for the paper-style translation-head score.

## Dev / Test Discipline

Use the official `2026 Czech-to-English Dev Set` to validate whether the promoted heads behave well in-domain.

Do not use the official `2026 Czech-to-English Test Set` for head discovery or promotion.

The `2025 Dev Set` is reasonable as a secondary check, but the 2026 dev set is the most direct target for current challenge conditions.

## Practical Recipe

1. Discover candidate heads on `CzEng 2.0`.
2. Re-score or stress-test the top heads on `Europarl` and `VoxPopuli cs->en`.
3. Freeze a compact promoted set only after it stays stable across those corpora.
4. Validate the promoted set on the official 2026 dev set inside the streaming stack.

## Command Skeleton

```bash
python assets/attention_heads/detect_translation_heads.py \
  --direction cs-en \
  --model google/gemma-4-E4B-it \
  --src-path /path/to/train.cs \
  --tgt-path /path/to/train.en \
  --dataset-name czeng2.0
```
